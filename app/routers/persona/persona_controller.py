from fastapi import APIRouter
from pydantic import BaseModel

from app.internal.exception.error_code import ControlledException, ErrorCode
from config.common.common_response import CommonResponse
from config.database.postgres_database import postgres_database
from config.database.milvus_database import milvus_database

router = APIRouter(prefix="/persona")

class PersonaRequest(BaseModel):
    """
    요약:
        /persona POST API를 이용하기 위해서 사용하는 Request Class

    설명:
        각 Attributes는 null를 전달하면 추가되지 않는다. (ego_id 제외)

    Attributes:
        ego_id(str): 페르소나가 생성 될 ego_id
        name(str|None): 에고의 이름
        age(int|None): 에고(사용자)의 나이
        mbti(str|None): 에고(사용자)의 mbti
    """
    ego_id: str
    name: str
    mbti: str
    interview: list[list[str]]

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
    if postgres_database.has_persona(ego_id=body.ego_id):
        raise ControlledException(ErrorCode.ALREADY_CREATED_EGO_ID)

    # Milvus에 중복되는 ego_id가 있는지 조회합니다.
    if milvus_database.has_partition(partition_name=body.ego_id):
        raise ControlledException(ErrorCode.ALREADY_CREATED_PARTITION)

    interview = build_interview_log(interview=body.interview)

    # NOTE 2. None이 아닌 값만 필터링해서 저장
    persona = {key: value for key, value in body.model_dump().items() if (key != "ego_id" or key != "interview") and value is not None}
    persona.update({"interview": interview})

    # NOTE 3. 에고 정보를 JSON으로 만들어 저장한다.
    postgres_database.insert_persona(ego_id= body.ego_id, persona=persona)

    # NOTE 4. Milvus Partition 생성
    # Milvus Database의 partition_name은 ego_id로 생성됩니다.
    milvus_database.create_partition(partition_name=body.ego_id)

    return CommonResponse(
        code=200,
        message="페르소나를 생성됐습니다!"
    )

class ToneRequest(BaseModel):
    """
    요약:
        /persona/tone POST API를 이용하기 위해서 사용하는 Request Class

    설명:
        각 Attributes는 필수이다. (not null)

    Attributes:
        ego_id(str): 말투가 생성 될 ego_id
        anger(str): 화남
        anxiety(str): 불안
        happiness(str): 행복
        neutrality(str): 평범
        sadness(str): 슬픔
    """
    ego_id: str
    anger: str
    anxiety: str
    happiness: str
    neutrality: str
    sadness: str

@router.post("/tone")
async def create_tone(body: ToneRequest)->CommonResponse:
    """
    요약:
        사용자의 말투를 저장하는 API

    설명:
        페르소나에 적용할 응답 말투를 지정한다.
    """
    # NOTE 1. 말투가 이미 존재하는지 확인한다.
    if postgres_database.has_tone(ego_id=body.ego_id):
        raise ControlledException(ErrorCode.ALREADY_CREATED_EGO_ID)

    # NOTE 2. PostgreSQL에 저장한다.
    tone = body.model_dump()
    postgres_database.insert_tone(ego_id=body.ego_id, tone=tone)

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