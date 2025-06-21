from pydantic import BaseModel


class ChatRequest(BaseModel):
    """
    요약:
        /chat/text Post API를 이용하기 위해서 사용하는 Request Class

    Attributes:
        message(str): 사용자가 질문한 메세지
        user_id(str): 질문한 사용자 ID
        ego_id(str): 답변할 Ego ID
    """
    message: str
    user_id: str
    ego_id: str