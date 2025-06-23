import os

import psycopg2
from dotenv import load_dotenv
from psycopg2 import DatabaseError

from app.internal.exception.error_code import ControlledException, ErrorCode
from config.common.common_database import CommonDatabase

load_dotenv()

POSTGRES_URI = os.getenv("POSTGRES_URI")
POSTGRES_DB_NAME = os.getenv("POSTGRES_DB_NAME")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

# TODO 1. 멀티스레드로 다중성 관리하기 # ConnectionPool
class PostgresDatabase(CommonDatabase):
    """
    PostgreSQL을 이용하기 위한 클래스

    psycopg2를 이용한 CommonDatabase 구현체
    """
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance.__connection = cls._instance._init_connection()
        return cls._instance

    def _init_connection(self):
        return psycopg2.connect(
            host=POSTGRES_URI,
            database=POSTGRES_DB_NAME,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            port=POSTGRES_PORT
        )

    def get_connection(self):
        return self.__connection

    def get_cursor(self):
        return self.__connection.cursor()

    def close(self):
        if self.__connection:
            self.__connection.close()

        if self.__class__._instance:
            self.__class__._instance = None

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