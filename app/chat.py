"""POST /chat  — conversational NLP assistant for the FSR record database.

Body (JSON):
    {
        "message": "What machines are non-operational?",
        "history": [                        # optional — last N turns
            {"role": "user",      "content": "Show me report A15273"},
            {"role": "assistant", "content": "...", "context_update": {...}}
        ]
    }

Response:
    {
        "text":           "...",            # markdown-formatted answer
        "data":           {...},            # structured payload for the UI
        "intent":         "search_status",
        "params":         {...},
        "context_update": {...},            # context to pass back in next turn
        "total_records":  10
    }
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .nlp.query_engine import answer

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None


@router.post("/chat")
def chat(req: ChatRequest) -> dict:
    if not req.message or not req.message.strip():
        raise HTTPException(400, "message must not be empty")
    if len(req.message) > 1000:
        raise HTTPException(400, "message too long (max 1000 chars)")
    return answer(req.message.strip(), history=req.history or [])
