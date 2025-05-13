import os

import psycopg2
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
        sql = f"INSERT INTO persona (name, persona) VALUES ('{persona_name}', '{json.dumps(persona_json)}')"
        self.__cursor.execute(sql)
        self.__database.commit()

    def update_persona(self, persona_id: int, persona_json: dict):
        sql = f"UPDATE persona SET persona = '{json.dumps(persona_json)}' WHERE persona_id = '{persona_id}'"
        self.__cursor.execute(sql)
        self.__database.commit()

    def select_persona_to_id(self, persona_id: int):
        sql = f"SELECT * FROM persona WHERE persona_id = '{persona_id}'"
        self.__cursor.execute(sql)
        return self.__cursor.fetchall()[0]

    @staticmethod
    def search_all_chat(user_id: str) -> list[str]:
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
            sql = f"SELECT * FROM chat_room WHERE DATE(last_chat_at) = CURRENT_DATE"
            cursor.execute(sql)
            chat_room_ids = [chat_room_id for chat_room_id, uid, egoId, last_chat_at, isDeleted in cursor.fetchall()]

            user_all_chat_room_log:list[str] = [] # 사용자의 모든 채팅방 대화 목록
            for chat_room_id in chat_room_ids:
                sql = f"SELECT * FROM chat_history WHERE chat_room_id = '{chat_room_id}'"
                cursor.execute(sql)

                # 사용자의 채팅방 대화 목록
                chat_room_log = "".join(
                    f"{'USER' if type == 'U' else 'AI'}: {content} at {chat_at}\n"
                    for chat_history_id, uid, chat_room_id, content, type, chat_at, is_deleted in cursor.fetchall()
                ).strip()

                user_all_chat_room_log.append(chat_room_log)
        except Exception as e:
            print("personalized 접속 실패")
            raise e
        finally:
            cursor.close()
            database.close()

        return user_all_chat_room_log

    def create_persona(self):
        sql = f"CREATE TABLE persona (persona_id SERIAL PRIMARY KEY, name VARCHAR(255) NOT NULL UNIQUE, persona JSON NOT NULL)"
        self.__cursor.execute(sql)
        self.__database.commit()

postgres_client = PostgresClient()