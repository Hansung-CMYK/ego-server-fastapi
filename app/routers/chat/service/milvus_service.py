from numpy import ndarray
from pymilvus import MilvusException

from app.internal.exception.error_code import ControlledException, ErrorCode
from app.internal.logger.logger import logger
from app.routers.chat.repository import (passages_repository,
                                         triplets_repository)
from app.routers.chat.service.parsed_sentence import ParsedSentence
from config.models.embedding_model import embedding_model

"""
DML
"""
def search_triplets(ego_id: str, field_name: str, datas: list[ndarray]) -> list[dict]:
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

        return triplets_repository.range_select_triplets(
            search_field=field_name,
            partition_names=[ego_id],
            output_fields=[
                "triplets_id",
                "passages_id",
                "subject",
                "object",
                "relation"
            ],
            data=datas,
        )[0]
    except MilvusException:
        logger.warning(f"\nMilvusException: {field_name}에서 해당 data로 entity 조회 실패. {field_name}로 조회를 생략합니다.\n")
        return []

def search_passages(ego_id: str, datas: list[int]) -> list[dict]:
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
        return passages_repository.select_passages_to_ids(
            partition_names=[ego_id],
            output_fields=[
                "passage"
            ],
            ids=datas
        )
    except MilvusException:
        logger.exception(f"\n\nMilvusException: passages에서 해당 ids로 entity 조회 실패\n""")
        raise ControlledException(ErrorCode.PASSAGE_NOT_FOUND)

def insert_messages(passage: str, single_sentences: list[str], ego_id: str):
    """
    요약:
        임베딩된 텍스트를 DB에 저장한다.

    Parameters:
        passage(str): 원문 정보
        single_sentences(list[str\]): 단일 문장으로 분리된 문장 리스트
        ego_id(str): 저장할 파티션 명
    """
    # 만약 partition이 생성되어 있지 않다면, 로그 생성 및 저장 안함
    if (not triplets_repository.has_triplets_partition(partition_name=ego_id)
            or not passages_repository.has_passages_partition(partition_name=ego_id)):
        raise ControlledException(ErrorCode.PARTITION_NOT_FOUND)

    # 문장을 삼중항으로 분리한다.
    parsed_sentences = [ParsedSentence(passage=passage, single_sentence=single_sentence) for single_sentence in
                        single_sentences]

    # NOTE 1. Passages에 값을 저장한다.
    passage_data = {
        "passage": passage,
        "embedded_passage": embedding_model.embedding(passage)[0],
        "is_fix": False,
    }

    # 실제 DB에 저장
    response = passages_repository.insert_passages(
        partition_name=ego_id,
        data=passage_data
    )
    passages_ids = response["ids"][0]  # res는 저장된 원문 ids 값

    # NOTE 2. triplets에 값을 저장한다.
    triplet_datas = []

    for index, parsed_sentence in enumerate(parsed_sentences):
        embedded_sentence = parsed_sentence.element_embedding()
        triplet_datas.append(
            {
                "passages_id": passages_ids,
                "subject": parsed_sentence.triplet[0],
                "object": parsed_sentence.triplet[1],
                "relation": parsed_sentence.relation,
                "embedded_subject": embedded_sentence["embedded_triplet"][0],
                "embedded_object": embedded_sentence["embedded_triplet"][1],
                "embedded_relation": embedded_sentence["embedded_relation"],
                "is_fix": False,
            }
        )

        triplets_repository.insert_triplets(
            partition_name=ego_id,
            data=triplet_datas
        )

def reset_collection(ego_id: str):
    """
    요약:
        collection에 있는 모든 삼중항 정보를 초기화하는 함수 캡스톤 시연에 활용하기 위함이다.

    설명:
        고정 텍스트가 아닌 채팅 내역을 삭제한다.

    Parameters:
        ego_id: 삭제될 파티션의 아이디(에고 아이디)
    """
    # id 값이 있는 모든 entity를 제거하는 함수이다.
    passages_repository.delete_passages(
        partition_name=ego_id,
        filter="is_fix == False"
    )
    triplets_repository.delete_triplets(
        partition_name=ego_id,
        filter="is_fix == False"
    )

"""
Partition
"""
def create_partition(partition_name: str):
    """
    요약:
        Milvus에 새로운 파티션을 추가하는 함수

    Parameters:
        partition_name: 추가할 파티션 명
    """
    passages_repository.create_passages_partition(partition_name=partition_name)
    triplets_repository.create_triplets_partition(partition_name=partition_name)

def delete_partition(partition_name: str):
    """
    파티션을 삭제하는 함수
    """
    passages_repository.drop_passages_partition(partition_name=partition_name)
    triplets_repository.drop_triplets_partition(partition_name=partition_name)

def has_partition(partition_name: str) -> bool:
    """
    요약:
        Milvus Database에 파티션이 존재하는지 확인하는 함수

    Parameters:
        partition_name(str): 존재하는지 확인할 파티션 명
    """
    return (
        passages_repository.has_passages_partition(partition_name=partition_name)
        or triplets_repository.has_triplets_partition(partition_name=partition_name)
    )