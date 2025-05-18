from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.common_response import CommonResponse
from app.models.postgres_database import postgres_database

router = APIRouter()

class PersonaRequest(BaseModel):
    """
    에고 페르소나 JSON을 만드는데 필요한 정보들이다.
    """
    ego_id: Optional[str] = None
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None

@router.post("/persona")
async def create_persona(body: PersonaRequest):
    # NOTE 1. 이미 존재하는 페르소나인지 조회한다.
    if postgres_database.already_persona(ego_id=body.ego_id):
        return CommonResponse(
            code=-1,
            message="이미 페르소나가 존재하는 에고입니다."
        )

    # NOTE 2. None이 아닌 값만 필터링해서 저장
    persona = {k: v for k, v in body.items() if v is not None}

    # NOTE 3. 에고 정보를 JSON으로 만들어 저장한다.
    postgres_database.insert_persona(persona=persona)

    return CommonResponse(
        code=200,
        message="페르소나를 생성됐습니다!"
    )