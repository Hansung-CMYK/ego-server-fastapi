import json
import threading

from config.common.common_database import CommonDatabase
from config.database.postgres_database import PostgresDatabase


class ToneRepository:
    """
    Postgres Tone을 호출하기 위한 Respository

    Tone과 관련된 SQL을 관리한다.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """
        Parameters:
            database(CommonDatabase): Database를 활용하기 위한 구현체
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance.database = kwargs.get("database") or PostgresDatabase()
                    cls._instance = instance
        return cls._instance

    def insert_tone(self, ego_id: str, tone: dict):
        """
        요약:
            말투를 추가하는 함수

        Parameters:
            ego_id(str): 추가할 에고의 아이디 * BE ego 테이블과 1대1 매핑되어야 한다.
            tone(dict): 추가할 말투 정보
        """
        self.database.execute_update(
            sql="INSERT INTO tone (ego_id, tone) VALUES (%s, %s)",
            values = (ego_id, json.dumps(tone),)
        )

    def has_tone(self, ego_id: str) -> bool:
        """
        요약:
            tone 테이블에 이미 ego_id가 존재하는지 확인하는 함수

        Parameters:
            ego_id(str): 존재하는지 확인힐 ego 아이디
        """
        result=self.database.execute_query(
            sql="SELECT * FROM tone WHERE ego_id = %s",
            values=(ego_id,)
        )
        if len(result) == 0: return False
        else: return True

    def delete_tone(self, ego_id: str):
        """
        모든 데이터를 제거하는 함수

        Parameters:
            ego_id: 제거할 ego_id
        """
        self.database.execute_update(
            sql="DELETE FROM tone WHERE ego_id = %s",
            values=(ego_id,)
        )

    def create_tone(self):
        """
        tone 테이블을 생성하는 함수
        """
        self.database.execute_update(
            sql="CREATE TABLE tone (ego_id VARCHAR(255) PRIMARY KEY, tone JSONB NOT NULL)"
        )

    def drop_tone(self):
        """
        tone 테이블을 제거하는 함수
        """
        self.database.execute_update(
            sql="DROP TABLE tone"
        )