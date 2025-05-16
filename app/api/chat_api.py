import os

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.common_response import CommonResponse
from app.services.chat_service import chat_stream
from app.services.session_config import SessionConfig

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    user_id: str
    ego_id: str

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

    return CommonResponse(
        code=200,
        message="answer success!",
        data=answer
    )

class AdminRequest(BaseModel):
    admin_id: str
    admin_password: str

ADMIN_ID = os.environ.get("POSTGRES_USER")
ADMIN_PASSWORD = os.environ.get("POSTGRES_PASSWORD")