from fastapi import APIRouter

from app.internal.exception.error_code import ControlledException, ErrorCode
from app.routers.tone import tone_service
from app.routers.tone.dto.tone_request import ToneRequest
from config.common.common_response import CommonResponse

router = APIRouter(prefix="/tone")


@router.post("/tone")
async def create_tone(body: ToneRequest)->CommonResponse:
    """
    요약:
        사용자의 말투를 저장하는 API

    설명:
        페르소나에 적용할 응답 말투를 지정한다.
    """
    # NOTE 1. 말투가 이미 존재하는지 확인한다.
    if tone_service.has_tone(ego_id=body.ego_id):
        raise ControlledException(ErrorCode.ALREADY_CREATED_EGO_ID)

    # NOTE 2. PostgreSQL에 저장한다.
    tone = body.model_dump()
    tone.pop("ego_id")
    tone_service.insert_tone(ego_id=body.ego_id, tone=tone)

    return CommonResponse(
        code=200,
        message="말투를 생성하였습니다."
    )