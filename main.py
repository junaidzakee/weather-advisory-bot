"""
FastAPI backend. Exposes:
  POST /chat        -> run one turn through the LangGraph agent
  GET  /new_session -> hand out a fresh session id
  /                 -> serves the chat frontend (frontend/index.html)
"""
import uuid
from dotenv import load_dotenv

load_dotenv()  # must run BEFORE importing graph, since graph -> llm reads env vars at call time

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from graph import build_graph

app = FastAPI()
_graph = build_graph()  # built once at startup, reused for every request


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    meta: dict


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.session_id}}
    result = _graph.invoke(
        {
            "user_question": req.message,
            "messages": [{"role": "user", "content": req.message}],
        },
        config=config,
    )
    return {"reply": result["final_answer"], "meta": result.get("response_meta", {})}


@app.get("/new_session")
def new_session():
    return {"session_id": str(uuid.uuid4())}


# Serve the frontend last, so it doesn't shadow the API routes above.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")