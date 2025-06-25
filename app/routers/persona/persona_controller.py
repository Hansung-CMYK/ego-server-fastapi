from fastapi import APIRouter

from app.internal.exception.error_code import ControlledException, ErrorCode
from app.routers.chat.service import milvus_service
from app.routers.persona import persona_service
from app.routers.persona.dto.persona_request import PersonaRequest
from app.routers.tone import tone_service
from config.common.common_response import CommonResponse
from config.external import hub_api

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
    if milvus_service.has_partition(partition_name=body.ego_id):
        raise ControlledException(ErrorCode.ALREADY_CREATED_PARTITION)


    # NOTE 2. None이 아닌 값만 필터링해서 저장
    # interview는 따로 관리
    interview = tone_service.interview_to_str(interview=body.interview)

    persona = persona_service.init_persona(body.model_dump())
    persona.update({"interview": interview})

    # NOTE 3. 에고 정보를 JSON으로 만들어 저장한다.
    persona_service.insert_persona(ego_id= body.ego_id, persona=persona)

    # NOTE 4. Milvus Partition 생성
    # Milvus Database의 partition_name은 ego_id로 생성됩니다.
    milvus_service.create_partition(partition_name=body.ego_id)

    return CommonResponse(
        code=200,
        message="페르소나를 생성됐습니다!"
    )

@router.patch("")
async def update_persona(body: PersonaRequest)->CommonResponse:
    """

    """
    # NOTE 1. 페르소나가 존재하는지 확인한다.
    if not persona_service.has_persona(ego_id=body.ego_id):
        raise ControlledException(ErrorCode.PERSONA_NOT_FOUND)

    persona = persona_service.select_persona_to_ego_id(ego_id=body.ego_id)[1]

    # NOTE 2. 페르소나 변경
    new_persona = body.model_dump()
    new_persona.pop("ego_id")
    persona.update({key: value for key, value in new_persona.values() if value is not None})

    # NOTE 3. 페르소나 저장
    persona_service.update_persona(ego_id=body.ego_id, persona=persona)

    return CommonResponse(
        code=200,
        message="페르소나를 변경하였습니다."
    )

@router.delete("/{user_id}")
async def delete_persona(user_id: str)->CommonResponse:
    """

    """
    # NOTE 1. ego_id가 존재하는지 확인한다.
    ego_id = hub_api.get_ego(user_id=user_id)["id"]

    # NOTE 2. 페르소나가 존재하는지 확인한다.
    if not persona_service.has_persona(ego_id=ego_id):
        raise ControlledException(ErrorCode.PERSONA_NOT_FOUND)

    # NOTE 3. 페르소나를 삭제한다.
    persona_service.delete_persona(ego_id=ego_id)

    return CommonResponse(
        code=200,
        message="페르소나를 삭제하였습니다."
    )
