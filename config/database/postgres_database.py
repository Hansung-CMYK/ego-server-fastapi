import os

import psycopg2
from dotenv import load_dotenv
from psycopg2 import DatabaseError

from app.internal.exception.error_code import ControlledException, ErrorCode
from config.common.common_database import CommonDatabase

load_dotenv()

# TODO 1. 멀티스레드로 다중성 관리하기 # ConnectionPool
class PostgresDatabase(CommonDatabase):
    """
    요약:
        PostgreSQL을 이용하기 위한 Client

        psycopg2를 이용한 CommonDatabase 구현체
    """
    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            with cls.__lock:
                if not cls.__instance:
                    cls.__instance = super().__new__(cls)
                    cls.__instance.__connection = cls.__instance.__init_connection()
        return cls.__instance

    def __init_connection(self):
        return psycopg2.connect(
            host=os.getenv("POSTGRES_URI"),
            database=os.getenv("POSTGRES_DB_NAME"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            port=os.getenv("POSTGRES_PORT")
        )

    def get_connection(self):
        return self.__connection

    def get_cursor(self):
        return self.__connection.cursor()

    def close(self):
        if self.__connection:
            self.__connection.close()

        if self.__class__.__instance:
            self.__class__.__instance = None

    def execute_update(self, sql: str, values: tuple=()):
        try:
            with self.get_cursor() as cursor:
                cursor.execute(sql, values)
            self.__connection.commit()
        except DatabaseError:
            self.__connection.rollback()
            raise ControlledException(ErrorCode.FAILURE_TRANSACTION)

    def execute_query(self, sql: str, values: tuple=()):
        try:
            with self.get_cursor() as cursor:
                cursor.execute(sql, values)
                return cursor.fetchall()
        except DatabaseError:
            raise ControlledException(ErrorCode.FAILURE_TRANSACTION)