from datetime import date

from pydantic import BaseModel


class DiaryRequest(BaseModel):
    """
    /diary POST API를 이용하기 위해서 사용하는 Request Class

    Attributes:
        user_id(str): 일기가 작성될 사용자 ID
        ego_id(int): 당일 에고 ID
        target_date(date): 일기에 포함될 날짜 * 대화를 조회할 시간
    """
    user_id: str
    ego_id: int
    target_date: date