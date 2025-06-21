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
    tone_service.insert_tone(ego_id=body.ego_id, tone=tone)

    return CommonResponse(
        code=200,
        message="말투를 생성하였습니다."
    )

def build_interview_log(interview: list[list[str]]) -> str:
    """
    인터뷰(2중 리스트)에서 첫 두 리스트를 교대로 이어붙여
    Java StringBuilder 로직과 동일한 로그 문자열을 반환합니다.

    Parameters
    ----------
    interview : list[list[str]]
        [[Q1, Q2, …], [A1, A2, …], …] 형태의 중첩 리스트

    Returns
    -------
    str
        Q1\nA1\nQ2\nA2\n … 형식의 문자열
    """
    if len(interview) < 2:           # 리스트가 2개 미만이면 빈 문자열 반환
        return ""

    first_list, second_list = interview[0], interview[1]

    lines: list[str] = []
    for q, a in zip(first_list, second_list):
        lines.append(q)
        lines.append(a)

    return "\n".join(lines)