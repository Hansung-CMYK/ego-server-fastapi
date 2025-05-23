import os

import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

from app.exception.exceptions import ControlledException, ErrorCode

load_dotenv()

class PostgresDatabase:
    # TODO: RollBack 쿼리 추가하기
    """
    요약:
        PostgreSQL을 이용하기 위한 Client

    Attributes:
        __database: 데이터베이스에 접근하는 connection이다.
        __cursor: SQL 쿼리를 수행하는 객체이다.
    """
    def __init__(self):
        self.__database = psycopg2.connect(
            host=os.getenv("POSTGRES_URI"),
            database=os.getenv("POSTGRES_DB_NAME"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            port=os.getenv("POSTGRES_PORT")
        )
        self.__cursor = self.__database.cursor()

    def __del__(self):
        self.__database.close()
        self.__cursor.close()

    def insert_persona(self, ego_id: str, persona: dict):
        """
        요약:
            페르소나를 추가하는 함수

        Parameters:
            ego_id(str): 추가할 에고의 아이디 * BE ego 테이블과 1대1 매핑되어야 한다.
            persona(dict): 추가할 페르소나 정보
        """
        sql = "INSERT INTO persona (ego_id, persona) VALUES (%s, %s)"
        self.__cursor.execute(sql, (ego_id, json.dumps(persona),))
        self.__database.commit()

    def update_persona(self, ego_id: str, user_persona: dict):
        """
        요약:
            기존 페르소나를 변경하는 함수

        Parameters:
            ego_id(str): 변경할 에고의 아이디
            user_persona(dict): 새로 저장할 사용자의 페르소나\
        """
        sql = "UPDATE persona SET persona = %s WHERE ego_id = %s"
        self.__cursor.execute(sql, (json.dumps(user_persona), ego_id,))
        self.__database.commit()

    def select_persona_to_ego_id(self, ego_id: str)->tuple:
        """
        요약:
            ego_id를 이용해 페르소나를 저장하는 함수

        Parameters:
            ego_id(str): 조회할 ego의 아이디

        Raises:
            PERSONA_NOT_FOUND: ego_id로 페르소나 조회 실패
        """
        sql = "SELECT * FROM persona WHERE ego_id = %s"
        self.__cursor.execute(sql, (ego_id,))
        result = self.__cursor.fetchall()

        if len(result) == 0: raise ControlledException(ErrorCode.PERSONA_NOT_FOUND)
        else: return result[0] # 페르소나 결과 반환

    def already_persona(self, ego_id: str) -> bool:
        """
        요약:
            persona 테이블에 이미 ego_id가 존재하는지 확인하는 함수

        Parameters:
            ego_id(str): 존재하는지 확인힐 ego 아이디
        """
        sql = "SELECT * FROM persona WHERE ego_id = %s"
        self.__cursor.execute(sql, (ego_id,))
        result = self.__cursor.fetchall()
        if len(result) == 0: return False
        else: return True

    def create_persona(self):
        """
        persona 테이블을 생성하는 함수
        """
        sql = "CREATE TABLE persona (ego_id INT PRIMARY KEY, persona JSON NOT NULL)"
        self.__cursor.execute(sql)
        self.__database.commit()

postgres_database = PostgresDatabase()