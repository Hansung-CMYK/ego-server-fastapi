import json

from app.internal.exception.error_code import ControlledException, ErrorCode
from config.common.common_database import CommonDatabase
from config.database.postgres_database import PostgresDatabase


class PersonaRepository:
    def __init__(self, database:CommonDatabase = PostgresDatabase()):
        self.database = database

    def insert_persona(self, ego_id: str, persona: dict):
        """
        요약:
            페르소나를 추가하는 함수

        Parameters:
            ego_id(str): 추가할 에고의 아이디 * BE ego 테이블과 1대1 매핑되어야 한다.
            persona(dict): 추가할 페르소나 정보
        """
        self.database.execute_update(
            sql="INSERT INTO persona (ego_id, persona) VALUES (%s, %s)",
            values=(ego_id, json.dumps(persona),)
        )

    def update_persona(self, ego_id: str, persona: dict):
        """
        요약:
            기존 페르소나를 변경하는 함수

        Parameters:
            ego_id(str): 변경할 에고의 아이디
            persona(dict): 새로 저장할 사용자의 페르소나\
        """
        self.database.execute_update(
            sql = "UPDATE persona SET persona = %s WHERE ego_id = %s",
            values=(json.dumps(persona), ego_id,)
        )

    def select_persona_to_ego_id(self, ego_id: str)->tuple:
        """
        요약:
            ego_id를 이용해 페르소나를 저장하는 함수

        Parameters:
            ego_id(str): 조회할 ego의 아이디

        Raises:
            PERSONA_NOT_FOUND: ego_id로 페르소나 조회 실패
        """
        result=self.database.execute_query(
                sql="SELECT * FROM persona WHERE ego_id = %s",
                values=(ego_id,)
        )

        if len(result) == 0:
            raise ControlledException(ErrorCode.PERSONA_NOT_FOUND)
        else:
            return result[0] # 페르소나 결과 반환

    def has_persona(self, ego_id: str) -> bool:
        """
        요약:
            persona 테이블에 이미 ego_id가 존재하는지 확인하는 함수

        Parameters:
            ego_id(str): 존재하는지 확인힐 ego 아이디
        """
        result=self.database.execute_query(
            sql="SELECT * FROM persona WHERE ego_id = %s",
            values=(ego_id,)
        )

        if len(result) == 0:
            return False
        else:
            return True

    def delete_persona(self, ego_id: str):
        """
        모든 데이터를 제거하는 함수
        """
        self.database.execute_update(
            sql="DELETE FROM persona WHERE ego_id = %s",
            values=(ego_id,)
        )

    def create_persona(self):
        """
        persona 테이블을 생성하는 함수
        """
        self.database.execute_update(
            sql = "CREATE TABLE persona (ego_id VARCHAR(255) PRIMARY KEY, persona JSONB NOT NULL)"
        )

    def drop_persona(self):
        """
        persona 테이블을 제거하는 함수
        """
        self.database.execute_update(
            sql = "DROP TABLE persona"
        )