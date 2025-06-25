from fastapi import APIRouter

from app.internal.exception.error_code import ControlledException, ErrorCode
from app.routers.tone import tone_service
from app.routers.tone.dto.tone_request import ToneRequest
from config.common.common_response import CommonResponse
from config.external import hub_api

router = APIRouter(prefix="/tone")


@router.post("")
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

@router.patch("")
async def update_tone(body: ToneRequest)->CommonResponse:
    """

    """
    # NOTE 1. 말투가 존재하는지 확인한다.
    if not tone_service.has_tone(ego_id=body.ego_id):
        raise ControlledException(ErrorCode.TONE_NOT_FOUND)

    tone = tone_service.select_tone_to_ego_id(ego_id=body.ego_id)[1]

    # NOTE 2. 말투 변경
    new_tone = body.model_dump()
    new_tone.pop("ego_id")
    tone.update({key: value for key, value in new_tone.values() if value is not None})

    # NOTE 3. 말투 저장
    tone_service.update_tone(ego_id=body.ego_id, tone=tone)

    return CommonResponse(
        code=200,
        message="말투를 변경하였습니다."
    )

@router.delete("/{user_id}")
async def delete_tone(user_id: str)->CommonResponse:
    """

    """
    # NOTE 1. ego_id가 존재하는지 확인한다.
    ego_id = hub_api.get_ego(user_id=user_id)["id"]

    # NOTE 2. 말투가 존재하는지 확인한다.
    if not tone_service.has_tone(ego_id=ego_id):
        raise ControlledException(ErrorCode.TONE_NOT_FOUND)

    # NOTE 3. 말투를 삭제한다.
    tone_service.delete_tone(ego_id=ego_id)

    return CommonResponse(
        code=200,
        message="말투를 삭제하였습니다."
    )
