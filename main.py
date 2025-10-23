import logging
import time
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel

from app.services import run_chat


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat.endpoint")

app = FastAPI(title="LangChain ShotGrid Chatbot")

class ChatReq(BaseModel):
    chatInput: str

@app.post("/chat")
def chat(req: ChatReq):
    start = time.perf_counter()
    thread_id = f"http-chat-{uuid4()}"
    result = run_chat(req.chatInput, thread_id=thread_id)
    duration = time.perf_counter() - start
    tool_count = len(result.get("tool_calls") or [])
    error = result.get("error")
    logger.info(
        "chat request thread=%s tools=%s error=%s duration=%.3fs",
        thread_id,
        tool_count,
        error,
        duration,
    )
    return {
        "reply": result.get("reply"),
        "tool_calls": result.get("tool_calls"),
        "api_response": result.get("tool_calls"),
        "error": error,
        "intent": None,
        "suggest_json": None,
    }
