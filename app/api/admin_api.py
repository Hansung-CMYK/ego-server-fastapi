import os

from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel

from app.api.common_response import CommonResponse
from app.exception.exceptions import ControlledException, ErrorCode
from app.models.database.milvus_database import milvus_database
from app.models.database.postgres_database import postgres_database
from app.services.chat.persona_store import KARINA_PERSONA, MYEONGJUN_PERSONA
from app.models.chat.main_llm import main_llm

load_dotenv()

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

@router.post("/reset/{user_id}/{ego_id}")
async def reset_ego(user_id:str, ego_id:str, body: AdminRequest)->CommonResponse:
    """
    요약:
        채팅 메모리와 데이터베이스 정보를 모두 삭제하는 명령어

    참고:
        milvus의 파티션은 제거하지 않습니다.

    Parameters:
        user_id(str): 초기화 할 사용자의 아이디
        ego_id(str): 초기화 할 사용자의 에고 아이디
        body(AdminRequest): 관리자 인증
    """
    if body.admin_id != ADMIN_ID or body.admin_password != ADMIN_PASSWORD:
        raise ControlledException(ErrorCode.INVALID_ADMIN_ID)

    if int(user_id.split("user_id_")[1]) != int(ego_id):
        return CommonResponse(
            code=500,
            message="에고 아이디와 유저 아이디가 일치하지 않습니다."
        )

    # NOTE 1. milvus_db의 파티션에 있는 정보들을 초기화한다.
    milvus_database.reset_collection(ego_id=ego_id)

    # NOTE 2. postgres의 페르소나 정보들을 삭제한다.
    postgres_database.delete_persona(ego_id=ego_id)

    # NOTE 3. postgres의 페르소나 정보들을 불러온다.
    postgres_database.insert_persona(
        ego_id=ego_id,
        persona=KARINA_PERSONA if ego_id == "5" else MYEONGJUN_PERSONA # 만약에 에고 아이디가 2가 아니면 명준 페르소나 삽입
    )

    # NOTE 4. 세션 메모리 기록을 삭제한다. (loop 돌면서 user_id 전부 검사)
    main_llm.reset_session_history(uid=user_id)

    # NOTE 5. 메모리 기록을 복구한다.
    # TODO 결정할 것: 기존 내용 넣기 or 그냥 안넣기

    return CommonResponse(
        code=200,
        message="reset ego success"
    )

@router.delete("/ego/{ego_id}")
async def delete_ego(ego_id: str, body:AdminRequest)->CommonResponse:
    """
    요약:
        방문객이 생성한 에고를 삭제하는 API

    Parameters:
        ego_id(str): 삭제할 에고의 아이디
        body(AdminRequest): 관리자 권한 소유 여부 확인
    """
    if body.admin_id != ADMIN_ID or body.admin_password != ADMIN_PASSWORD:
        raise ControlledException(ErrorCode.INVALID_ADMIN_ID)

    # NOTE 1. PostgreSQL persona 테이블 삭제
    postgres_database.delete_persona(ego_id=ego_id)

    # NOTE 2. PostgreSQL tone 테이블 삭제
    postgres_database.delete_tone(ego_id=ego_id)

    # NOTE 3. Milvus Partition 삭제
    milvus_database.delete_partition(ego_id=ego_id)

    return CommonResponse(
        code=200,
        message="delete ego success"
    )

