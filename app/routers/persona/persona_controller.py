from fastapi import APIRouter

from app.internal.exception.error_code import ControlledException, ErrorCode
from app.routers.persona import persona_service
from app.routers.persona.dto.persona_request import PersonaRequest
from app.routers.tone.tone_controller import build_interview_log
from config.common.common_response import CommonResponse
from config.database.milvus_database import milvus_database

router = APIRouter(prefix="/persona")


@router.post("")
async def create_persona(body: PersonaRequest)->CommonResponse:
    """
    요약:
        페르소나를 생성하는 API

    설명:
        페르소나 생성과 함께 Milvus에 지식그래프가 생성된다.

    Parameters:
        body(PersonaRequest): 페르소나 생성에 필요한 인자의 모음
            * ego_id, name, age, gender, mbti를 Attribute로 갖는다.
    """
    # NOTE 1. 이미 존재하는 페르소나인지 조회한다.
    # PostgreSQL에 중복되는 ego_id가 있는지 조회합니다.
    if persona_service.has_persona(ego_id=body.ego_id):
        raise ControlledException(ErrorCode.ALREADY_CREATED_EGO_ID)

    # Milvus에 중복되는 ego_id가 있는지 조회합니다.
    if milvus_database.has_partition(partition_name=body.ego_id):
        raise ControlledException(ErrorCode.ALREADY_CREATED_PARTITION)

    interview = build_interview_log(interview=body.interview)

    # NOTE 2. None이 아닌 값만 필터링해서 저장
    persona = {key: value for key, value in body.model_dump().items() if (key != "ego_id" or key != "interview") and value is not None}
    persona.update({"interview": interview})

    # NOTE 3. 에고 정보를 JSON으로 만들어 저장한다.
    persona_service.insert_persona(ego_id= body.ego_id, persona=persona)

    # NOTE 4. Milvus Partition 생성
    # Milvus Database의 partition_name은 ego_id로 생성됩니다.
    milvus_database.create_partition(partition_name=body.ego_id)

    return CommonResponse(
        code=200,
        message="페르소나를 생성됐습니다!"
    )