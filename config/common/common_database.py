import threading
from abc import ABC, abstractmethod


class CommonDatabase(ABC):
    """
    프로젝트에서 데이터베이스 구현을 위해 준수해야 할 필수 인터페이스

    관계형 데이터베이스를 구현할 시, 꼭 다음 함수를 이용해주세요.

    Attributes:
        _instance: 싱글턴 인스턴스입니다.
        _lock: 싱글턴을 구현하기 위한 동기화 Flag 객체입니다.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """
        싱글턴 구현을 위한 함수입니다.

        인스턴스를 호출할 땐 CommonDatabase() 혹은 상속 객체를 호출해주세요.
        """
        pass

    @abstractmethod
    def _init_connection(self):
        """
        connection을 생성하는 함수입니다.

        Returns: Connection object
        """
        pass

    @abstractmethod
    def get_connection(self):
        """
        데이터베이스 Connection을 반환하는 함수입니다.

        Returns: Connection object
        """
        pass

    @abstractmethod
    def get_cursor(self):
        """
        Connection의 Cursor를 반환하는 함수입니다.

        Returns: Cursor object
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        해당 클래스의 인스턴스를 종료하는 함수입니다.

        Connection과 (싱글턴이라면) Instance를 close해주세요.
        """
        pass

    @abstractmethod
    def execute_update(self, sql: str, values: tuple=()) -> None:
        """
        create, insert, update, delete sql 구현에 사용하는 함수입니다.

        Parameters:
            sql(str): 구현할 SQL입니다.
            values(tuple): SQL에 포함될 데이터입니다.
        """
        pass

    @abstractmethod
    def execute_query(self, sql: str, values: tuple=()):
        """
        select sql 구현에 사용하는 함수입니다.

        Parameters:
            sql(str): 구현할 SQL입니다.
            values(tuple): SQL에 포함될 데이터입니다.
        """
        pass
