# app/api/ollama_api.py
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse

from app.services.chat_history_service import get_chat_history_prompt
from app.services.graph_rag_service import get_rag_prompt
from app.services.ollama_service import chat_full, chat_stream

router = APIRouter()

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@router.post(
    "/ollama",
    response_model=ChatResponse,
    summary="Ollama 모델에 메시지 전달하고 전체 응답 받기",
    tags=["ollama"],
)
async def http_ollama(body: ChatRequest) -> JSONResponse:
    """
    요청 예시:
    {
      "message": "너의 이름은 무엇이니"
    }
    """
    resp = chat_full(body.message)
    return JSONResponse({"response": resp})

ego_name = "ego"
session_id = "1234"
@router.websocket("/ws/ollama")
async def ws_ollama(ws: WebSocket):
    await ws.accept()
    try:
        raw = await ws.receive_text()
        payload = json.loads(raw)
        prompt = payload.get("message")
        if not prompt:
            await ws.send_text(json.dumps({"error": "message 필요"}))
            return
        rag_prompt = get_rag_prompt(ego_name=ego_name, user_speak=prompt)
        chat_history_prompt = get_chat_history_prompt(session_id=f"{ego_name}@{session_id}")
        for chunk in chat_stream(prompt):
            await ws.send_text(json.dumps({"chunk": chunk}))
        await ws.send_text(json.dumps({"done": True}))
    except WebSocketDisconnect:
        return

