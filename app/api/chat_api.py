import json
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse

from app.models.main_llm_model import main_llm
from app.services.chat_service import chat_full, chat_stream, save_graphdb, save_persona
from app.services.persona_store import persona_store
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

    store_keys = main_llm.get_store_keys()  # 모든 채팅방의 세션 키값을 불러온다.

    for session_id in store_keys: # 모든 세션(채팅방)을 순회한다.
        ego_id, user_id = session_id.split("@")

        # 세션 정보로 해당 채팅방의 대화 내역을 불러온다.
        session_history = main_llm.get_human_messages_in_memory(session_id=session_id)

        # NOTE 1. 대화 내역을 기반으로 페르소나 저장
        # TODO: 유저 단위가 아닌 채팅방 단위이므로, 현재 매우 비효율적인 로직이다. (user_id를 기반으로 묶어 저장하는 것이 정석)
        # TODO: 채팅내역 받아와서 정리하기
        save_persona(ego_id=ego_id, session_history=session_history)

        # NOTE 2. 대화 내역을 기반으로 Graph Database 저장
        save_graphdb(ego_id=ego_id, session_history=session_history)

        # NOTE 3. 채팅 내역 초기화
        # Graph Database에 중복된 문장을 저장하지 않기 위해 기존 대화 내역을 제거한다.
        main_llm.reset_session_history(session_id=session_id)

    # NOTE 4. 기존에 저장되어있던 PersonaStore.store 정보들 초기화(remove unused data)
    persona_store.remove_all_persona()