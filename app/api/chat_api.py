import os
import traceback, logging, base64

from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

from app.api.common_response import CommonResponse
from app.exception.exceptions import ControlledException, ErrorCode
from app.services.chatting.chat_service import chat_stream
from app.services.session_config import SessionConfig

from app.models.image.image_descriptor import ImageDescriptor

logger = logging.getLogger("chat_api")

router = APIRouter(prefix="/chat")

class ChatRequest(BaseModel):
    """
    요약:
        /chat/text Post API를 이용하기 위해서 사용하는 Request Class

    Attributes:
        message(str): 사용자가 질문한 메세지
        user_id(str): 질문한 사용자 ID
        ego_id(str): 답변할 Ego ID
    """
    message: str
    user_id: str
    ego_id: str

@router.post("/text")
async def create_ai_chat(body: ChatRequest)->CommonResponse:
    """
    요약:
        페르소나와 대화하는 API

    설명:
        Kafka와 웹소켓을 이용하지 않고 직접적으로 채팅합니다.
        main_llm에서 전체 채팅이 반환되면 response가 반환됩니다.

    Parameters:
        body(ChatRequest): 사용자의 답변에 필요한 인자의 모음
            * message, user_id, ego_id를 Attributes로 갖는다.
    """
    # NOTE: 에고 채팅 생성
    ai_message:str = ""
    for chunk in chat_stream( # main_llm의 답변이 chunk 단위로 전달됩니다.
        user_message=body.message,
        config=SessionConfig(body.user_id, body.ego_id)
    ):
        ai_message += chunk # 모든 chunk를 answer에 연결합니다.

    return CommonResponse( # 성공 시, Code 200을 반환합니다. 예외처리는 chat_stream을 참고해주세요.
        code=200,
        message="answer success!",
        data=ai_message
    )

@router.post("/image")
async def ollama_image(
    file: UploadFile = File(..., description="이미지 파일"),
    user_id: str    = Form(..., description="사용자 ID"),
    ego_id:  str    = Form(..., description="페르소나 ID"),
):  
    try :
        session_config = SessionConfig(user_id, ego_id)
        raw_bytes = await file.read()

        b64_image = base64.b64encode(raw_bytes).decode("utf-8")

        image_description = ImageDescriptor.invoke(b64_image=b64_image)
        ImageDescriptor.store(image_description, session_config)
        
        return CommonResponse(
            code=200,
            message="answer success!",
            data=image_description
        )
    except Exception:
        raise ControlledException(ErrorCode.IMAGE_DESCRIPTION_ERROR)
    
class AdminRequest(BaseModel):
    """
    요약:
        /admin POST API를 사용하기 위해서 사용하는 Request Class

    Attributes:
        admin_id(str): 관리자 계정의 ID
        admin_password(str): 관리자 계정의 Password
    """
    admin_id: str
    admin_password: str

# 따로 관리자 계정의 ID가 정해지지 않아서, PostgreSQL Root 계정의 정보를 활용한다.
ADMIN_ID = os.environ.get("POSTGRES_USER")
ADMIN_PASSWORD = os.environ.get("POSTGRES_PASSWORD")