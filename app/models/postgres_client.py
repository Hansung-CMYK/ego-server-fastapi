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

    def select_persona_to_name(self, persona_name: str):
        sql = f"SELECT * FROM persona WHERE name = '{persona_name}'"
        self.__cursor.execute(sql)
        return self.__cursor.fetchall()[0]

    def select_persona_to_id(self, persona_id: int):
        sql = f"SELECT * FROM persona WHERE persona_id = {persona_id}"
        self.__cursor.execute(sql)
        return self.__cursor.fetchall()[0]

    def search_all(self):
        sql = f"SELECT * FROM persona"
        self.__cursor.execute(sql)
        return self.__cursor.fetchall()

    def create_persona(self):
        sql = f"CREATE TABLE persona (persona_id SERIAL PRIMARY KEY, name VARCHAR(255) NOT NULL UNIQUE, persona JSON NOT NULL)"
        self.__cursor.execute(sql)
        self.__database.commit()

postgres_client = PostgresClient()