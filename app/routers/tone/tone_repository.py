import json

from app.internal.exception.error_code import ControlledException, ErrorCode
from config.database.postgres_database import PostgresDatabase

"""
Postgres Tone을 호출하기 위한 Respository

Tone과 관련된 SQL을 관리한다.
"""

database = PostgresDatabase()

"""
DDL
"""
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

def has_tone()->bool:
    return 1 == database.execute_update(
        sql="SELECT 1 FROM Information_schema.tables WHERE table_name = 'tone' AND table_schema = 'postgres'"
    )

"""
DML
"""
def select_tone_to_ego_id(ego_id: str)->tuple:
    """
    요약:
        ego_id를 이용해 말투를 조회하는 함수

    Parameters:
        ego_id(str): 조회할 ego의 아이디

    Raises:
        PERSONA_NOT_FOUND: ego_id로 말투 조회 실패
    """
    result=database.execute_query(
            sql="SELECT * FROM persona WHERE ego_id = %s",
            values=(ego_id,)
    )

    if len(result) == 0:
        raise ControlledException(ErrorCode.TONE_NOT_FOUND)
    else:
        return result[0] # 페르소나 결과 반환

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

def update_tone(ego_id: str, tone: dict):
    database.execute_update(
        sql="update tone set tone = %s where ego_id = %s",
        values= (ego_id, json.dumps(tone),)
    )

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

"""
Another
"""
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

    if len(result) == 0:
        return False
    else:
        return True