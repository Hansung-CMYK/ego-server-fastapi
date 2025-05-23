import logging
import os

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("admin_api")

router = APIRouter(prefix="/admin")

# 따로 관리자 계정의 ID가 정해지지 않아서, PostgreSQL Root 계정의 정보를 활용한다.
ADMIN_ID = os.environ.get("POSTGRES_USER")
ADMIN_PASSWORD = os.environ.get("POSTGRES_PASSWORD")

class AdminRequest(BaseModel):
    """
    요약:
        /admin POST API를 사용하기 위해서 사용하는 Request Class

    Attributes:
        admin_id(str): 관리자 계정의 ID
        admin_password(str): 관리자 계정의 Password
    """
    admin_id: str
    admin_password: str