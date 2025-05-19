import os

import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

from app.exception.exceptions import ControlledException, ErrorCode

load_dotenv()

class PostgresDatabase:
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
        sql = "INSERT INTO persona (ego_id, persona) VALUES (%s, %s)"
        self.__cursor.execute(sql, (ego_id, json.dumps(persona),))
        self.__database.commit()

    def update_persona(self, ego_id: str, persona_json: dict):
        sql = "UPDATE persona SET persona = %s WHERE ego_id = %s"
        self.__cursor.execute(sql, (json.dumps(persona_json), ego_id,))
        self.__database.commit()

    def select_persona_to_id(self, ego_id: str):
        sql = "SELECT * FROM persona WHERE ego_id = %s"
        self.__cursor.execute(sql, (ego_id,))
        result = self.__cursor.fetchall()

        if len(result) == 0: raise ControlledException(ErrorCode.PERSONA_NOT_FOUND)
        else: return result[0] # 페르소나 결과 반환

    def already_persona(self, ego_id: str) -> bool:
        sql = "SELECT * FROM persona WHERE ego_id = %s"
        self.__cursor.execute(sql, (ego_id,))
        result = self.__cursor.fetchall()
        if len(result) == 0: return False
        else: return True

    def create_persona(self):
        sql = "CREATE TABLE persona (ego_id INT PRIMARY KEY, persona JSON NOT NULL)"
        self.__cursor.execute(sql)
        self.__database.commit()

postgres_database = PostgresDatabase()