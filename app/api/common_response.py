from typing import TypeVar, Generic

T = TypeVar('T')

class CommonResponse(Generic[T]):
    def __init__(self, code: int, message: str, data: T = None):
        self.code = code
        self.message = message
        self.data = data