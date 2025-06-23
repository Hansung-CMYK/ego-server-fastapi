import json
import threading

from config.common.common_database import CommonDatabase
from config.database.postgres_database import PostgresDatabase

"""
Postgres Tone을 호출하기 위한 Respository

Tone과 관련된 SQL을 관리한다.
"""

database = PostgresDatabase()

def insert_tone(ego_id: str, tone: dict):
    """
    요약:
        말투를 추가하는 함수

    Parameters:
        ego_id(str): 추가할 에고의 아이디 * BE ego 테이블과 1대1 매핑되어야 한다.
        tone(dict): 추가할 말투 정보
    """
    database.execute_update(
        sql="INSERT INTO tone (ego_id, tone) VALUES (%s, %s)",
        values = (ego_id, json.dumps(tone),)
    )

def has_tone(ego_id: str) -> bool:
    """
    요약:
        tone 테이블에 이미 ego_id가 존재하는지 확인하는 함수

    Parameters:
        ego_id(str): 존재하는지 확인힐 ego 아이디
    """
    result=database.execute_query(
        sql="SELECT * FROM tone WHERE ego_id = %s",
        values=(ego_id,)
    )
    if len(result) == 0: return False
    else: return True

def delete_tone(ego_id: str):
    """
    모든 데이터를 제거하는 함수

    Parameters:
        ego_id: 제거할 ego_id
    """
    database.execute_update(
        sql="DELETE FROM tone WHERE ego_id = %s",
        values=(ego_id,)
    )

def create_tone():
    """
    tone 테이블을 생성하는 함수
    """
    database.execute_update(
        sql="CREATE TABLE tone (ego_id VARCHAR(255) PRIMARY KEY, tone JSONB NOT NULL)"
    )

def drop_tone():
    """
    tone 테이블을 제거하는 함수
    """
    database.execute_update(
        sql="DROP TABLE tone"
    )