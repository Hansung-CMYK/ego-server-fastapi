import json
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse

from app.services.chat_service import chat_full, chat_stream, save_graphdb, save_persona
from app.services.session_config import SessionConfig

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    user_id: str
    ego_id: str

class ChatResponse(BaseModel):
    code: int
    message: str
    response: str

# @router.post(
#     "/ollama",
#     response_model=ChatResponse,
#     summary="Ollama 모델에 메시지 전달하고 전체 응답 받기",
#     tags=["ollama"],
# )
# async def http_ollama(body: ChatRequest) -> JSONResponse:
#     """
#     요청 예시:
#     {
#       "message": "너의 이름은 무엇이니"
#     }
#     """
#     resp = chat_full(body.message)
#     return JSONResponse({"response": resp})
#
# @router.websocket("/ws/ollama")
# async def ws_ollama(ws: WebSocket):
#     await ws.accept()
#     try:
#         raw = await ws.receive_text()
#         payload = json.loads(raw)
#         prompt = payload.get("message")
#         if not prompt:
#             await ws.send_text(json.dumps({"error": "message 필요"}))
#             return
#         for chunk in chat_stream(prompt):
#             await ws.send_text(json.dumps({"chunk": chunk}))
#         await ws.send_text(json.dumps({"done": True}))
#     except WebSocketDisconnect:
#         return

@router.post("/chat/ollama_chat")
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

    return ChatResponse(
        code=200,
        message="answer success!",
        response=answer
    )

class AdminRequest(BaseModel):
    admin_id: str
    admin_password: str

ADMIN_ID = os.environ.get("POSTGRES_USER")
ADMIN_PASSWORD = os.environ.get("POSTGRES_PASSWORD")

@router.post("/admin/save_persona_metadata")
async def save_persona_metadata(body: AdminRequest):
    # NOTE 0. 민감 API 임시 로그인 처리
    if body.admin_id != ADMIN_ID or body.admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin credentials")

    # NOTE 1. 대화 내역을 기반으로 페르소나 저장
    save_persona()

    # NOTE 2. 대화 내역을 기반으로 Graph Database 저장
    save_graphdb()