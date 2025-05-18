import os

from dotenv import load_dotenv
from numpy import ndarray
from pymilvus import MilvusClient, MilvusException
import logging

from app.exception.exception_handler import ControlledException
from app.exception.exceptions import ErrorCode
from app.models.parsed_sentence import ParsedSentence

# .env 환경 변수 추출
load_dotenv()
URI = os.getenv('URI')

class MilvusDatabase:
    """
    Milvus를 이용하기 위한 Client이다.

    각종 함수에 활용된다.
    Attibutes:
        milvus_client:
    """
    def  __init__(self):
        self.__milvus_client = MilvusClient(
            uri=URI,
            token="root:Milvus"
        )

    def search_triplets_to_milvus(self, ego_id: str, field_name: str, datas: list[ndarray]) -> list[dict]:
        """
            triplets 컬렉션에서 연관된 삼중항을 조회하는 함수이다.

            Parameters:
                ego_id (str): 조회할 파티션 이름
                field_name (str): 조회할 필드 이름 (벡터 필드)
                datas (list[ndarray]): 검색하고자 하는 벡터 데이터

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
                        "radius": 0.7
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
            logging.warning(f"MilvusException: {field_name}에서 해당 data로 entity 조회 실패. {field_name}로 조회를 생략합니다.")
            return []

    def search_passages_to_milvus(self, ego_id: str, datas: list[int]) -> list[dict]:
        """
            주어진 ID 리스트를 기준으로 passages 컬렉션에서 원문 데이터를 조회한다.

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
        except MilvusException as e:
            logging.error(f"""
                MilvusException: passages에서 해당 ids로 entity 조회 실패
                에러 발생 데이터: {datas}
                예외 내용: {e}
            """)
            raise ControlledException(ErrorCode.PASSAGE_NOT_FOUND)

    def insert_messages_into_milvus(
            self,
            splited_messages: list[str],
            ego_id: str
    ):
        """
        임베딩된 텍스트를 DB에 저장한다.

        :param splited_messages: 단일 문장으로 분리된 문장 리스트
        :param ego_id: 저장할 파티션 명
        """
        # TODO: [계정 생성 연동 이전이라 생성되는 문제] 업데이트 시, 제거할 것
        # 만약 partition이 생성되어있지 않다면, 새 파티션 생성
        if not self.__milvus_client.has_partition(collection_name="triplets", partition_name=ego_id):
            self.__milvus_client.create_partition(collection_name="passages", partition_name=ego_id)
            self.__milvus_client.create_partition(collection_name="triplets", partition_name=ego_id)

        # 문장을 삼중항으로 Parsing한다.
        parsed_sentences = [ParsedSentence(splited_message) for splited_message in splited_messages]

        # NOTE 1. Passages에 값을 저장한다.
        for speak in parsed_sentences:
            embedded_speak = speak.embedding()
            passage_data = {
                "passage": speak.passage,
                "embedded_passage": embedded_speak["embedded_passage"]
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
            for index, triplet in enumerate(speak.triplets) :
                triplet_datas.append(
                    {
                        "passages_id": passages_ids,
                        "subject": triplet[0],
                        "object": triplet[1],
                        "relation": speak.relations[index],
                        "embedded_subject": embedded_speak["embedded_triplets"][index][0],
                        "embedded_object": embedded_speak["embedded_triplets"][index][1],
                        "embedded_relation": embedded_speak["embedded_relations"][index]
                    }
                )

            self.__milvus_client.insert(
                collection_name="triplets",
                partition_name=ego_id,
                data=triplet_datas
            )

milvus_database = MilvusDatabase()