from dotenv import load_dotenv
import os
import certifi
import json
import uuid

from pathlib import Path

import uvicorn

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage,
)

from agent import get_agent

from database import (
    init_db,
    save_chat_message,
    get_chat_history,
    create_or_update_conversation,
    list_conversations,
)

from rag import add_document_to_rag
from tools import set_current_thread_id


# ==========================================
# Environment and paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Support both the correctly spelled folder and the folder name
# used earlier in this project ("templats").
TEMPLATE_DIR = BASE_DIR / "templates"

if not TEMPLATE_DIR.is_dir():
    legacy_template_dir = BASE_DIR / "templats"
    if legacy_template_dir.is_dir():
        TEMPLATE_DIR = legacy_template_dir

if not TEMPLATE_DIR.is_dir():
    raise RuntimeError(
        "Templates folder was not found. Create 'templates' "
        "or rename the existing 'templats' folder."
    )


# ==========================================
# App setup
# ==========================================

app = FastAPI(title="Wizora")

templates = Jinja2Templates(
    directory=str(TEMPLATE_DIR)
)

UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

init_db()


# ==========================================
# Home page
# ==========================================

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


# ==========================================
# Conversations
# ==========================================

@app.get("/conversations")
async def conversations():
    items = list_conversations()

    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
            }
            for item in items
        ]
    }


@app.get("/history/{thread_id}")
async def history(thread_id: str):
    messages = get_chat_history(thread_id)

    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
            }
            for msg in messages
        ]
    }


# ==========================================
# Upload documents
# ==========================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...),
):
    try:
        allowed_extensions = {
            ".pdf",
            ".docx",
            ".txt",
            ".md",
            ".py",
            ".csv",
        }

        filename = Path(
            (file.filename or "uploaded_file").replace("\\", "/")
        ).name

        suffix = Path(filename).suffix.lower()

        if suffix not in allowed_extensions:
            return JSONResponse(
                {
                    "success": False,
                    "message": (
                        "Unsupported file type. Upload PDF, DOCX, TXT, "
                        "MD, PY, or CSV."
                    ),
                },
                status_code=400,
            )

        if not thread_id.strip():
            return JSONResponse(
                {
                    "success": False,
                    "message": "Thread ID is required.",
                },
                status_code=400,
            )

        file_id = str(uuid.uuid4())
        safe_filename = filename.replace(" ", "_")
        file_path = UPLOAD_DIR / f"{file_id}_{safe_filename}"

        content = await file.read()

        if not content:
            return JSONResponse(
                {
                    "success": False,
                    "message": "The uploaded file is empty.",
                },
                status_code=400,
            )

        with open(file_path, "wb") as output_file:
            output_file.write(content)

        create_or_update_conversation(
            thread_id,
            "Uploaded document",
        )

        result = add_document_to_rag(
            file_path=str(file_path),
            thread_id=thread_id,
        )

        return JSONResponse(
            {
                "success": True,
                "message": (
                    f"Uploaded {result['filename']} "
                    f"and created {result['chunks']} chunks."
                ),
            }
        )

    except Exception as error:
        return JSONResponse(
            {
                "success": False,
                "message": str(error),
            },
            status_code=500,
        )

    finally:
        await file.close()


# ==========================================
# Streaming helpers
# ==========================================

def sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def should_stream_chunk(chunk, metadata) -> bool:
    metadata = metadata or {}

    node_name = str(
        metadata.get("langgraph_node", "")
    ).lower()

    if "tool" in node_name:
        return False

    if isinstance(chunk, ToolMessage):
        return False

    if not isinstance(chunk, (AIMessage, AIMessageChunk)):
        return False

    if getattr(chunk, "tool_calls", None):
        return False

    if getattr(chunk, "invalid_tool_calls", None):
        return False

    additional_kwargs = getattr(
        chunk,
        "additional_kwargs",
        {},
    ) or {}

    if additional_kwargs.get("tool_calls"):
        return False

    return True


def extract_text_from_chunk(chunk) -> str:
    content = getattr(chunk, "content", "")

    if not content:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

                elif isinstance(item.get("text"), str):
                    text_parts.append(item["text"])

                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])

        return "".join(text_parts)

    return ""


# ==========================================
# Chat streaming
# ==========================================

@app.post("/chat/stream")
async def chat_stream(request: Request):
    try:
        request_data = await request.json()

    except Exception:
        return JSONResponse(
            {"error": "Invalid JSON body."},
            status_code=400,
        )

    if not isinstance(request_data, dict):
        return JSONResponse(
            {"error": "JSON body must be an object."},
            status_code=400,
        )

    user_message = request_data.get("message", "")
    thread_id = request_data.get("thread_id", "default")

    selected_model = request_data.get("model") or os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite",
    )

    if not isinstance(user_message, str) or not user_message.strip():
        return JSONResponse(
            {"error": "Message is required."},
            status_code=400,
        )

    if not isinstance(thread_id, str) or not thread_id.strip():
        return JSONResponse(
            {"error": "Thread ID is required."},
            status_code=400,
        )

    agent = get_agent(selected_model)

    create_or_update_conversation(
        thread_id,
        user_message,
    )

    save_chat_message(
        thread_id,
        "user",
        user_message,
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    def event_generator():
        final_answer = ""

        try:
            set_current_thread_id(thread_id)

            inputs = {
                "messages": [
                    HumanMessage(content=user_message)
                ]
            }

            for chunk, metadata in agent.stream(
                inputs,
                config=config,
                stream_mode="messages",
            ):
                if not should_stream_chunk(chunk, metadata):
                    continue

                token = extract_text_from_chunk(chunk)

                if token:
                    final_answer += token
                    yield sse_data({"token": token})

            if final_answer.strip():
                save_chat_message(
                    thread_id,
                    "assistant",
                    final_answer,
                )

            yield sse_data({"done": True})

        except Exception as error:
            yield sse_data({"error": str(error)})
            yield sse_data({"done": True})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8080,
        reload=True,
    )
