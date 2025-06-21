from pydantic import BaseModel


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