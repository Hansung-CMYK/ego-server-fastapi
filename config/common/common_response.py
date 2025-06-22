from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

# Generic Class를 생성한다.
T = TypeVar('T')

class CommonResponse(BaseModel, Generic[T]):
    """
    FastAPI에서 공통으로 사용하는 Response 클래스

    Attributes:
         code(int): 반환 코드
         message(str): 반환 코드의 의미
         data(T): 전달할 데이터 * default: None
    """
    code: int
    message: str
    data: Optional[T] = None