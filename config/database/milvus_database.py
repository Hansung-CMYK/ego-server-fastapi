import os
import threading

from dotenv import load_dotenv
from numpy import ndarray
from pymilvus import CollectionSchema, MilvusClient
from pymilvus.milvus_client import IndexParams

from config.models.embedding_model import embedding_model

load_dotenv()

# .env 환경 변수 추출
MILVUS_URI = os.getenv('MILVUS_URI')

"""
임베딩 모델의 최대 차원 수를 명시한 상수이다. 
"""
embedding_dim = len(embedding_model.embedding(["임베딩 모델 차원 수 측정용"])[0])

class MilvusDatabase:
    """
    벡터 데이터베이스(Milvus)에서 공통적으로 이용하는 함수를 관리하는 클래스

    pymilvus를 이용해 싱글턴으로 구현하였습니다.

    Attributes:
        _instance: 싱글턴 인스턴스입니다.
        _lock: 싱글턴을 구현하기 위한 동기화 Flag 객체입니다.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls.__connection = cls._instance._init_connection()
        return cls._instance

    @staticmethod
    def _init_connection():
        """
        connection을 생성하는 함수입니다.
        """
        return MilvusClient(
            uri=MILVUS_URI,
            token="root:Milvus"
        )

    def get_connection(self):
        """
        데이터베이스 Connection을 반환하는 함수

        현재 생성되지 않은 milvus_client 로직을 수행하기 위한 백도어도 수행
        """
        return self.__connection

    def close(self):
        """
        해당 클래스의 인스턴스를 종료하는 함수

        Connection과 (싱글턴이라면) Instance를 close해주세요.
        """
        if self.__connection:
            self.__connection.close()

        if self.__class__._instance:
            self.__class__._instance = None

    """
    DDL
    """
    def create_collection(self, collection_name:str, schema:CollectionSchema, index_params:IndexParams):
        """
        콜렉션(collection)을 생성하는 함수

        이를 수행하기 위해선 사전에 schema와 index_params를 우선 할당할 것

        Parameters:
            collection_name(str): 생성할 콜렉션 명
            schema(CollectionSchema): 콜렉션의 스키마(fields)
            index_params(IndexParams): 스키마의 인덱스
        """
        return self.get_connection().create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            dimension=embedding_dim
        )

    def drop_collection(self, collection_name:str):
        """
        콜렉션(collection)을 삭제하는 함수

        Parameters:
            collection_name(str): 삭제할 콜렉션 명
        """
        return self.get_connection().drop_collection(
            collection_name=collection_name
        )

    def has_collection(self, collection_name:str):
        self.get_connection().has_collection(collection_name=collection_name)

    def create_schema(self):
        self.get_connection().create_schema(auto_id=True, enable_dynamic_field=True)

    def prepare_index_params(self):
        self.get_connection().prepare_index_params()

    """
    DML
    """
    def select_all(self, collection_name: str, partition_names: list[str], output_fields:list[str], filter:str="id >= 0"):
        """
        콜렉션 레코드를 전체 조회하는 함수

        Parameters:
            collection_name(str): 조회할 콜렉션 명
            partition_names(list[str]): 조회할 파티션 묶음
            output_fields(list[str]): 반환받고 싶은 field 명
            filter(str): 전체 검색을 하는 조건
        """
        return self.get_connection().query(
            collection_name=collection_name,
            partition_names=partition_names,
            output_fields=output_fields,
            filter=filter
        )

    def select_passages_to_ids(self, collection_name: str, partition_names: list[str], output_fields: list[str], ids: int|list[int]):
        """
        콜렉션 레코드를 id를 통해 조회하는 함수

        Parameters:
            collection_name(str): 조회할 콜렉션 명
            partition_names(list[str]): 조회할 파티션 묶음
            output_fields(list[str]): 반환받고 싶은 field 명
            ids(int|list[int]): 조회할 id
        """
        return self.get_connection().get(
            collection_name=collection_name,
            partition_names=partition_names,
            output_fields=output_fields,
            ids=ids
        )

    def range_select(self, collection_name: str, search_field: str, partition_names: list[str],
                              output_fields: list[str], data: ndarray|list[ndarray], radius: float = 0.6):
        """
        datas와 인접한 벡터를 가진 콜렉션 레코드를 조회하는 함수

        Parameters:
            collection_name(str): 조회할 콜렉션 명
            partition_names(list[str]): 조회할 파티션 묶음
            output_fields(list[str]): 반환받고 싶은 field 명
            search_field(str): 인접 벡터를 구할 벡터 필드
            data(ndarray|list[ndarray]): 인접 벡터를 구할 기준 벡터(임베딩 텍스트)
            radius(float): 레코드 유사도 범위(높을수록 유사한 것 *0.0~1.0)
        """
        return self.get_connection().search(
            collection_name=collection_name,
            partition_names=partition_names,
            output_fields=output_fields,
            search_params={
                "metric_type": "COSINE",
                "params": {
                    "radius": radius
                }
            },
            anns_field=search_field,
            data=data
        )

    def insert(self, collection_name:str, partition_name:str, data:dict|list[dict]):
        """
        콜렉션 레코드를 추가하는 함수

        Parameters:
            collection_name(str): 조회할 콜렉션 명
            partition_name(str): 조회할 파티션 묶음
            data(dict|list[dict]): 인접 벡터를 구할 기준 벡터(임베딩 텍스트)
        """
        return self.get_connection().insert(
            collection_name=collection_name,
            partition_name=partition_name,
            data=data
        )

    def delete(self, collection_name: str, partition_name: str, filter:str):
        """
        콜렉션 레코드를 삭제하는 함수

        Parameters:
            partition_name(str): 삭제할 파티션 묶음
            filter(str): 삭제할 원문의 조건
        """
        return self.get_connection().delete(
            collection_name=collection_name,
            partition_name=partition_name,
            filter=filter
        )

    """
    Partition
    """
    def create_partition(self, collection_name:str, partition_name:str):
        """
        콜렉션에 새로운 파티션을 추가하는 함수

        Parameters:
            collection_name(str): 추가할 콜렉션 명
            partition_name(str): 추가할 파티션의 명
        """
        return self.get_connection().create_partition(
            collection_name=collection_name,
            partition_name=partition_name
        )

    def drop_partition(self, collection_name:str, partition_name:str):
        """
        콜렉션에 새로운 파티션을 제거하는 함수

        Parameters:
            collection_name(str): 제거할 콜렉션 명
            partition_name(str): 제거할 파티션 명
        """
        return self.get_connection().drop_partition(
            collection_name=collection_name,
            partition_name=partition_name
        )

    def has_partition(self, collection_name:str, partition_name:str):
        """
        콜렉션에 이미 파티션이 존재하는지 확인하는 함수

        Parameters:
            collection_name(str): 확인할 콜렉션 명
            partition_name(str): 확인할 파티션 명
        """
        return self.get_connection().has_partition(
            collection_name=collection_name,
            partition_name=partition_name
        )

    def load_partitions(self, collection_name:str, partition_names:str|list[str]):
        """
        Milvus에 파티션을 불러오는 함수

        불러오지 않은 파티션은 사용하지 못한다.

        Parameters:
            collection_name(str): 해당 파티션이 있는 콜렉션
            partition_names(list[str]): 불러올 파티션 명
        """
        return self.get_connection().load_partitions(
            collection_name=collection_name,
            partition_names=partition_names
        )

    def get_load_state(self, collection_name:str, partition_name:str):
        """
        Milvus에 로딩된 파티션을 확인하는 함수

        Parameters:
            collection_name(str): 해당 파티션이 있는 콜렉션
            partition_name(str): 확인할 파티션 명
        """
        return self.get_connection().get_load_state(
            collection_name=collection_name,
            partition_name=partition_name
        )

    def release_partitions(self, collection_name:str, partition_names:list[str]):
        """
        Milvus에 파티션을 해제하는 함수

        불러오지 않은 파티션은 사용하지 못한다.

        Parameters:
            collection_name(str): 해당 파티션이 있는 콜렉션
            partition_names(list[str]): 해제할 파티션 명
        """
        return self.get_connection().release_partitions(
            collection_name=collection_name,
            partition_names=partition_names
        )