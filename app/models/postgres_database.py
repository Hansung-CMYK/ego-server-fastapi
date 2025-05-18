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

    def insert_persona(self, persona: dict):
        sql = "INSERT INTO persona (persona) VALUES (%s)"
        self.__cursor.execute(sql, (json.dumps(persona),))
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

    @staticmethod
    def search_all_chat(user_id: str, target_time: datetime):
        """
        user_id에 맞는 사용자의 대화 내역을 불러오는 함수
        """
        try:
            database = psycopg2.connect(
                host=os.getenv("POSTGRES_URI"),
                database="personalized-data",
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                port=os.getenv("POSTGRES_PORT"),
                options=f"-c search_path={user_id}"
            )
            cursor = database.cursor()

            # TODO: 현재 날짜 설정이 자정을 넘어가면 연산 방식을 바꿔야함. ex) 현재시간부터 -24시간 이내

            # 오늘 한번이라도 대화한 채팅방의 정보를 조회한다.
            start_time = target_time - timedelta(hours=24)
            end_time = target_time

            sql = "SELECT * FROM chat_room WHERE last_chat_at BETWEEN %s AND %s"
            cursor.execute(sql, (start_time, end_time))
            chat_room_ids = [chat_room_id for chat_room_id, uid, egoId, last_chat_at, isDeleted in cursor.fetchall()]

            user_all_chat_room_log:list[list[str]] = [] # 사용자의 모든 채팅방 대화 목록
            for chat_room_id in chat_room_ids:
                sql = "SELECT * FROM chat_history WHERE chat_room_id = %s"
                cursor.execute(sql, (chat_room_id, ))

                # 사용자의 채팅방 대화 목록
                chat_room_log = [f"{'USER' if type == 'U' else 'AI'}: {content} at {chat_at}" for chat_history_id, uid, chat_room_id, content, type, chat_at, is_deleted, message_hash in cursor.fetchall()]

                user_all_chat_room_log.append(chat_room_log)
        except psycopg2.OperationalError:
            raise ControlledException(ErrorCode.POSTGRES_ACCESS_DENIED)
        except psycopg2.ProgrammingError:
            raise ControlledException(ErrorCode.INVALID_SQL_ERROR)
        finally:
            cursor.close()
            database.close()

        return user_all_chat_room_log

    def create_persona(self):
        sql = "CREATE TABLE persona (ego_id INT PRIMARY KEY, persona JSON NOT NULL)"
        self.__cursor.execute(sql)
        self.__database.commit()

postgres_database = PostgresDatabase()