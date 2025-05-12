import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse

from app.services.ollama_service import chat_full, chat_stream
from app.services.session_config import SessionConfig

router = APIRouter()

class ChatRequest(BaseModel):
    ego_name: str
    user_id: str
    ego_id: str

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
        for chunk in chat_stream(prompt):
            await ws.send_text(json.dumps({"chunk": chunk}))
        await ws.send_text(json.dumps({"done": True}))
    except WebSocketDisconnect:
        return

@router.post("/ws/ollama_chat")
async def ollama_chat(body: ChatRequest):
    """
    채팅을 통한 페르소나 대화

    user_id와 ego_id를 통해 채팅방에 접근할 수 있다.
    """
    answer:str = ""
    for chunk in chat_stream(
        prompt=body.message,
        config=SessionConfig(body.user_id, body.ego_id)
    ):
        answer += chunk

    return answer