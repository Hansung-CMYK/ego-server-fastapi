from pydantic import BaseModel


class PersonaRequest(BaseModel):
    """
    요약:
        /persona POST API를 이용하기 위해서 사용하는 Request Class

    설명:
        각 Attributes는 null를 전달하면 추가되지 않는다. (ego_id 제외)

    Attributes:
        ego_id(str): 페르소나가 생성 될 ego_id
        name(str|None): 에고의 이름
        mbti(str|None): 에고(사용자)의 mbti
    """
    ego_id: str
    name: str
    mbti: str
    interview: list[list[str]]