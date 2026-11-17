import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import certifi


# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from tools import tools


# Project settings
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_PATH = DATA_DIR / "langgraph_checkpoints.sqlite"

DEFAULT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

ALLOWED_MODELS = {
    "gemini-3.5-flash-lite",
}

ALLOWED_MODELS.add(DEFAULT_MODEL)


# Wizora instructions
SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named Wizora.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest and current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall saved information when useful.
7. Use the calculator for mathematical questions.

Rules:
- For latest news, current events, prices, updates,
  and other time-sensitive information, use Tavily Search.
- If the user asks about uploaded documents,
  use search_uploaded_documents.
- If the user asks you to remember something,
  use remember_this.
- If the user asks about saved information,
  use recall_memory.
- Use calculator for mathematical questions.
- Do not invent information.
- Be clear, helpful, and concise.
"""


def normalize_model_name(model_name: str | None) -> str:
    """
    Validate the selected model.
    Return the default model if the selection is invalid.
    """

    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name


def build_agent(model_name: str):
    """
    Build a LangGraph agent for the selected Gemini model.
    """

    selected_model = normalize_model_name(model_name)

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.3,
        streaming=True
    )

    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):

        messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ] + state["messages"]

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response]
        }

    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesState)

    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "chatbot")

    workflow.add_conditional_edges(
        "chatbot",
        tools_condition
    )

    workflow.add_edge("tools", "chatbot")

    conn = sqlite3.connect(
        str(DATABASE_PATH),
        check_same_thread=False
    )

    checkpointer = SqliteSaver(conn)

    return workflow.compile(
        checkpointer=checkpointer
    )


_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    """
    Return an existing agent or create a new one.
    Each selected model has its own cached agent.
    """

    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]
