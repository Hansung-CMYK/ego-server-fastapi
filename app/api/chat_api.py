import os
import traceback, logging, base64

from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

from app.api.common_response import CommonResponse
from app.services.chat_service import chat_stream
from app.services.session_config import SessionConfig
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from app.models.image_descriptor import ImageDescriptor

logger = logging.getLogger("chat_api")

router = APIRouter(prefix="/chat")

class ChatRequest(BaseModel):
    message: str
    user_id: str
    ego_id: str

@router.post("/text")
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

@router.post("/image")
async def ollama_image(
    file: UploadFile = File(..., description="이미지 파일"),
    user_id: str    = Form(..., description="사용자 ID"),
    ego_id:  str    = Form(..., description="페르소나 ID"),
):  
    try :
        session_id:str = f"{ego_id}@{user_id}"
        raw_bytes = await file.read()

        b64_str = base64.b64encode(raw_bytes).decode("utf-8")

        data_uri = f"data:{file.content_type};base64,{b64_str}"
        
        return CommonResponse(
            code=200,
            message="answer success!",
            data=ImageDescriptor().invoke(data_uri)
        )
    
    except Exception:
        logger.error(f"이미지 처리 에러 {traceback.format_exc()}")
        return None

class AdminRequest(BaseModel):
    admin_id: str
    admin_password: str

ADMIN_ID = os.environ.get("POSTGRES_USER")
ADMIN_PASSWORD = os.environ.get("POSTGRES_PASSWORD")