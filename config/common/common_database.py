import threading
from abc import ABC, abstractmethod


class CommonDatabase(ABC):
    """
    프로젝트에서 데이터베이스 구현을 위해 준수해야 할 필수 인터페이스

    데이터베이스 드라이버를 구현할 시, 꼭 다음 함수를 이용해주세요.

    Attributes:
        _instances: _instances: 자식 클래스들의 싱글턴 인스턴스입니다.
            - _template(list): 각 prompt를 연결할 객체이다. 추가 TEMPLATE는 이 객체에 .append() 할 것
        _lock: 싱글턴을 구현하기 위한 동기화 Flag 객체입니다.
    """
    _instances: dict[type, 'CommonDatabase'] = {}
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """
        싱글턴 구현을 위한 함수입니다.

        인스턴스를 호출할 땐 CommonDatabase() 혹은 상속 객체를 호출해주세요.
        """
        if cls not in cls._instances:  # 1차 검사
            with cls._lock:
                if cls not in cls._instances:  # 2차 검사
                    instance = super().__new__(cls)
                    cls._instances[cls] = instance
        return cls._instances[cls]

    def __init__(self, *args, **kwargs) -> None:
        # 한 번만 실행되도록
        if getattr(self, "_initialized", False):
            return
        self._connection = self._init_connection()
        self._initialized = True

    @abstractmethod
    def _init_connection(self):
        """
        connection을 생성하는 함수입니다.
        """
        pass

    @abstractmethod
    def get_connection(self):
        """
        데이터베이스 Connection을 반환하는 함수입니다.
        """
        pass

    @abstractmethod
    def get_cursor(self):
        """
        Connection의 Cursor를 반환하는 함수입니다.
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
