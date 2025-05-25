import os

from dotenv import load_dotenv
from numpy import ndarray
from pymilvus import MilvusClient, MilvusException

from app.exception.exception_handler import ControlledException
from app.exception.exceptions import ErrorCode
from app.models.parsed_sentence import ParsedSentence
from app.logger.logger import logger

# .env 환경 변수 추출
load_dotenv()
MILVUS_URI = os.getenv('MILVUS_URI')

class MilvusDatabase:
    """
    요약:
        Milvus를 이용하기 위한 Client

    Attributes:
        __milvus_client(MilvusClient): milvus database에 접근 할 수 있도록 도와주는 객체
    """
    def __init__(self):
        self.__milvus_client = MilvusClient(
            uri=MILVUS_URI,
            token="root:Milvus"
        )

    def get_milvus_client(self)->MilvusClient:
        """
        현재 생성되지 않은 milvus_client 로직을 수행하기 위한 함수
        """
        return self.__milvus_client

    def search_triplets(self, ego_id: str, field_name: str, datas: list[ndarray]) -> list[dict]:
        """
        요약:
            triplets 컬렉션에서 연관된 삼중항을 조회하는 함수

        Parameters:
            ego_id(str): 조회할 파티션 이름
            field_name(str): 조회할 필드 이름 (벡터 필드)
            datas(list[ndarray]): 검색하고자 하는 벡터 데이터

        Returns:
            list[dict]: 검색된 삼중항 정보들의 리스트.
                        각 항목은 dict 형태이며, 검색 결과가 없으면 빈 리스트를 반환한다.

                        [
                            {
                                distance: FLOAT,
                                entity: {
                                    "triplets_id": INT64,
                                    "passages_id": INT64,
                                    "subject": str,
                                    "object": str,
                                }
                                ...,
                            },
                            ...
                        ]

        Raises:
            MilvusException: Milvus 조회 중 오류가 발생한 경우, 예외를 로깅하고 빈 리스트를 반환한다. 주로 연관 데이터가 없을 때 사용한다.
        """
        try:
            if len(datas) == 0: return []  # 아무 의미없는 값 조회 시, 예외처리

            return self.__milvus_client.search(
                collection_name="triplets",
                anns_field=field_name,
                partition_names=[ego_id],
                data=datas,
                search_params={
                    "metric_type": "COSINE",
                    "params": {
                        "radius": 0.5
                    }
                },
                output_fields=[
                    "triplets_id",
                    "passages_id",
                    "subject",
                    "object",
                    "relation"
                ],
            )[0]
        except MilvusException:
            logger.warning(f"\nMilvusException: {field_name}에서 해당 data로 entity 조회 실패. {field_name}로 조회를 생략합니다.\n")
            return []

    def search_passages(self, ego_id: str, datas: list[int]) -> list[dict]:
        """
        요약:
            주어진 ID 리스트를 기준으로 passages 컬렉션에서 원문 데이터를 조회

        Parameters:
            ego_id (int): 조회할 파티션 이름
            datas (list[int]): 검색하고자 하는 passages_id 리스트

        Returns:
            list[dict]: 검색된 passage 원문 리스트.
                        각 항목은 dict 형태이며, 실패 시 EntityNotFound 예외가 발생한다.

                        [
                            {
                                "passage": str,
                            },
                            ...
                        ]

        Raises:
            MilvusException: Milvus 조회 중 내부 오류가 발생한 경우 예외를 로깅한 후 전달한다.
            EntityNotFound: 해당 ID로 passage가 존재하지 않을 경우 발생한다.
        """
        try:
            return self.__milvus_client.get(
                collection_name="passages",
                partition_names=[ego_id],
                ids=datas,
                output_fields=[
                    "passage"
                ],
            )
        except MilvusException:
            logger.exception(f"\n\nMilvusException: passages에서 해당 ids로 entity 조회 실패\n""")
            raise ControlledException(ErrorCode.PASSAGE_NOT_FOUND)

    def insert_messages(
            self,
            splited_messages: list[str],
            ego_id: str
    ):
        """
        요약:
            임베딩된 텍스트를 DB에 저장한다.

        Parameters:
            splited_messages(list[str]): 단일 문장으로 분리된 문장 리스트
            ego_id(str): 저장할 파티션 명
        """
        # 만약 partition이 생성되어 있지 않다면, 로그 생성 및 저장 안함
        if (not self.__milvus_client.has_partition(collection_name="triplets", partition_name=ego_id)
            or not self.__milvus_client.has_partition(collection_name="passages", partition_name=ego_id)):
                raise ControlledException(ErrorCode.PARTITION_NOT_FOUND)

        # 문장을 삼중항으로 분리한다.
        parsed_sentences = [ParsedSentence(splited_message) for splited_message in splited_messages]

        # NOTE 1. Passages에 값을 저장한다.
        for parsed_sentence in parsed_sentences:
            embedded_speak = parsed_sentence.embedding()
            passage_data = {
                "passage": parsed_sentence.passage,
                "embedded_passage": embedded_speak["embedded_passage"],
                "is_fix": False,
            }

            # 실제 DB에 저장
            res = self.__milvus_client.insert(
                collection_name="passages",
                partition_name=ego_id,
                data=[passage_data]
            )
            passages_ids = res["ids"][0] # res는 저장된 원문 ids 값

            # NOTE 2. triplets에 값을 저장한다.
            triplet_datas = []
            for index, triplet in enumerate(parsed_sentence.triplets) :
                triplet_datas.append(
                    {
                        "passages_id": passages_ids,
                        "subject": triplet[0],
                        "object": triplet[1],
                        "relation": parsed_sentence.relations[index],
                        "embedded_subject": embedded_speak["embedded_triplets"][index][0],
                        "embedded_object": embedded_speak["embedded_triplets"][index][1],
                        "embedded_relation": embedded_speak["embedded_relations"][index],
                        "is_fix": False,
                    }
                )

            self.__milvus_client.insert(
                collection_name="triplets",
                partition_name=ego_id,
                data=triplet_datas
            )

    def has_partition(self, partition_name: str) -> bool:
        """
        요약:
            Milvus Database에 파티션이 존재하는지 확인하는 함수

        Parameters:
            partition_name(str): 존재하는지 확인할 파티션 명
        """
        return (
            self.__milvus_client.has_partition(collection_name="passages",partition_name=partition_name)
            or self.__milvus_client.has_partition(collection_name="triplets",partition_name=partition_name)
        )

    def create_partition(self, partition_name: str):
        """
        요약:
            Milvus에 새로운 파티션을 추가하는 함수

        Parameters:
            partition_name: 추가할 파티션 명
        """
        self.__milvus_client.create_partition(
            collection_name="passages",
            partition_name=partition_name
        )
        self.__milvus_client.create_partition(
            collection_name="triplets",
            partition_name=partition_name
        )

    def reset_collection(self, ego_id: str):
        """
        요약:
            collection에 있는 모든 삼중항 정보를 초기화하는 함수 캡스톤 시연에 활용하기 위함이다.

        설명:
            고정 텍스트가 아닌 채팅 내역을 삭제한다.

        Parameters:
            ego_id: 삭제될 파티션의 아이디(에고 아이디)
        """
        # id 값이 있는 모든 entity를 제거하는 함수이다.
        self.__milvus_client.delete(
            collection_name="passages",
            partition_name=ego_id,
            filter="is_fix == False"
        )
        self.__milvus_client.delete(
            collection_name="triplets",
            partition_name=ego_id,
            filter="is_fix == False"
        )

    def delete_partition(self, ego_id: str):
        """
        파티션을 삭제하는 함수
        """
        self.__milvus_client.release_partitions(collection_name="passages",partition_names=[ego_id])
        self.__milvus_client.release_partitions(collection_name="triplets",partition_names=[ego_id])

milvus_database = MilvusDatabase()