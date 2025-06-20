from abc import ABC, abstractmethod

class CommonDatabase(ABC):
    @abstractmethod
    def get_connection(self): pass

    @abstractmethod
    def get_cursor(self): pass

    @abstractmethod
    def close(self): pass

    @abstractmethod
    def execute_update(self, sql: str, values: tuple=()): pass

    @abstractmethod
    def execute_query(self, sql: str, values: tuple=()): pass
