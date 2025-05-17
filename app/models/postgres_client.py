import os

import psycopg2
from datetime import datetime
from dotenv import load_dotenv
import json

load_dotenv()

class PostgresClient:
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

    def insert_persona(self, persona_name: str, persona_json: dict):
        sql = "INSERT INTO persona (name, persona) VALUES (%s, %s)"
        self.__cursor.execute(sql, (persona_name, json.dumps(persona_json), ))
        self.__database.commit()

    def update_persona(self, persona_id: int, persona_json: dict):
        sql = "UPDATE persona SET persona = %s WHERE persona_id = %s"
        self.__cursor.execute(sql, (json.dumps(persona_json), persona_id, ))
        self.__database.commit()

    def select_persona_to_id(self, persona_id: int):
        sql = "SELECT * FROM persona WHERE persona_id = %s"
        self.__cursor.execute(sql, (persona_id, ))
        result = self.__cursor.fetchall()

        if len(result) == 0: return [
            persona_id,
            "카리나",
            {
                "name": "카리나",
                "age": 25,
                "gender": "여자",
                "mbti": "ENTP",
                "updated_at": datetime.now().isoformat()
            }
        ] # 페르소나 조회 실패 시, 예외처리 # TODO: 사용자 생성 업데이트 되면, 제거할 것
        else: return result[0] # 페르소나 결과 반환

    @staticmethod
    def search_all_chat(user_id: str):
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
            sql = "SELECT * FROM chat_room WHERE DATE(last_chat_at) = CURRENT_DATE"
            cursor.execute(sql)
            chat_room_ids = [chat_room_id for chat_room_id, uid, egoId, last_chat_at, isDeleted in cursor.fetchall()]

            user_all_chat_room_log:list[list[str]] = [] # 사용자의 모든 채팅방 대화 목록
            for chat_room_id in chat_room_ids:
                sql = "SELECT * FROM chat_history WHERE chat_room_id = %s"
                cursor.execute(sql, (chat_room_id, ))

                # 사용자의 채팅방 대화 목록
                chat_room_log = [f"{'USER' if type == 'U' else 'AI'}: {content} at {chat_at}" for chat_history_id, uid, chat_room_id, content, type, chat_at, is_deleted, message_hash in cursor.fetchall()]

                user_all_chat_room_log.append(chat_room_log)
        except Exception as e:
            print("personalized 접속 실패")
            raise e
        finally:
            cursor.close()
            database.close()

        return user_all_chat_room_log

    def create_persona(self):
        sql = "CREATE TABLE persona (persona_id SERIAL PRIMARY KEY, name VARCHAR(255) NOT NULL UNIQUE, persona JSON NOT NULL)"
        self.__cursor.execute(sql)
        self.__database.commit()

postgres_client = PostgresClient()