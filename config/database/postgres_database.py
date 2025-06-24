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
    def _init_connection(self):
        return psycopg2.connect(
            host=POSTGRES_URI,
            database=POSTGRES_DB_NAME,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            port=POSTGRES_PORT
        )

    def get_connection(self):
        return self._connection

    def get_cursor(self):
        return self._connection.cursor()

    def close(self):
        if self._connection:
            self._connection.close()

        self.__class__._instances.pop(self.__class__, None)

    def execute_update(self, sql: str, values: tuple=()):
        try:
            with self.get_cursor() as cursor:
                cursor.execute(sql, values)
            self._connection.commit()
        except DatabaseError:
            self._connection.rollback()
            raise ControlledException(ErrorCode.FAILURE_TRANSACTION)

    def execute_query(self, sql: str, values: tuple=()):
        try:
            with self.get_cursor() as cursor:
                cursor.execute(sql, values)
                return cursor.fetchall()
        except DatabaseError:
            raise ControlledException(ErrorCode.FAILURE_TRANSACTION)