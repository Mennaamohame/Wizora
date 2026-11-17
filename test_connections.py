
import os
from pathlib import Path
import requests
from dotenv import load_dotenv
from langsmith import Client

load_dotenv(Path(__file__).resolve().parent / ".env")

# 1. Test Gemini
try:
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
    )

    response = requests.post(
        url,
        headers={"x-goog-api-key": os.getenv("GOOGLE_API_KEY", "")},
        json={
            "contents": [
                {"parts": [{"text": "Say hello to Wizora."}]}
            ]
        },
        timeout=30
    )
    response.raise_for_status()

    print("Gemini: SUCCESS")

except Exception as e:
    print("Gemini: FAILED")
    print(type(e).__name__, str(e)[:200])


# 2. Test Tavily
try:
    response = requests.post(
        "https://api.tavily.com/search",
        headers={
            "Authorization": f"Bearer {os.getenv('TAVILY_API_KEY', '')}"
        },
        json={
            "query": "What is LangGraph?",
            "search_depth": "basic",
            "max_results": 1
        },
        timeout=30
    )
    response.raise_for_status()

    results = response.json().get("results", [])
    print(f"Tavily: SUCCESS ({len(results)} result)")

except Exception as e:
    print("Tavily: FAILED")
    print(type(e).__name__, str(e)[:200])


# 3. Test LangSmith (optional)
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

if not langsmith_api_key or langsmith_api_key == "your_langsmith_api_key":
    print("LangSmith: SKIPPED (no API key configured)")
else:
    try:
        client = Client(
            api_key=langsmith_api_key,
            api_url=os.getenv("LANGSMITH_ENDPOINT")
        )

        next(iter(client.list_projects(limit=1)), None)
        print("LangSmith: SUCCESS")

    except Exception as e:
        print("LangSmith: FAILED")
        print(type(e).__name__, str(e)[:200])
