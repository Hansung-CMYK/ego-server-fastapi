from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class ContentType(str, Enum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"


class ChatMessage(BaseModel):
    chatRoomId: int
    from_: str = Field(..., alias="from")
    to: str
    content: str
    contentType: ContentType
    messageHash: Optional[str] = None
    isDeleted: bool = False
    mcpEnabled: bool = False

    class Config:
        populate_by_name = True
