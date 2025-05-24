import os

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.common_response import CommonResponse
from app.exception.exceptions import ControlledException, ErrorCode
from app.models.database.milvus_database import milvus_database
from app.models.database.postgres_database import postgres_database

router = APIRouter(prefix="/admin")

# 따로 관리자 계정의 ID가 정해지지 않아서, PostgreSQL Root 계정의 정보를 활용한다.
ADMIN_ID = os.environ.get("POSTGRES_USER")
ADMIN_PASSWORD = os.environ.get("POSTGRES_PASSWORD")

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

@router.post("/reset/{ego_id}")
async def reset_ego(ego_id:str, body: AdminRequest)->CommonResponse:
    """
    요약:
        채팅 메모리와 데이터베이스 정보를 모두 삭제하는 명령어

    참고:
        milvus의 파티션은 제거하지 않습니다.
    """
    if body.admin_id != ADMIN_ID or body.admin_password != ADMIN_PASSWORD:
        raise ControlledException(ErrorCode.INVALID_ADMIN_ID)

    # milvus_database 리셋
    milvus_database.reset_collection(ego_id=ego_id)

    # postgres_database 리셋
    postgres_database.delete_persona(ego_id=ego_id)

    return CommonResponse(
        code=200,
        message="delete success"
    )

