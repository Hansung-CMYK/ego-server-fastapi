import numpy as np
from numpy import ndarray
from collections import defaultdict
from scipy.sparse import csr_matrix
from pymilvus import MilvusException
import logging

from app.exception.entity_not_found import EntityNotFound


def search_triplets_to_milvus(partition_name: str, field_name: str, datas: list[ndarray]) -> list[dict]:
    """
        triplets 컬렉션에서 연관된 삼중항을 조회하는 함수이다.

        Parameters:
            partition_name (str): 조회할 파티션 이름
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
        if len(datas) == 0: return [] # 아무 의미없는 값 조회 시, 예외처리

        from app.models.singleton import database_client # 순환 호출 문제로 인해 내부 import

        return database_client.milvus_client.search(
            collection_name="triplets",
            anns_field=field_name,
            partition_names=[partition_name],
            data=datas,
            search_params= {
                "metric_type": "COSINE",
                "params": {
                    "radius": 0.4
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
    except MilvusException as e:
        logging.error(f"::Error Exception(MilvusException):: {field_name}에서 해당 data로 entity 조회 실패")
        logging.error(f"::에러 발생 내용:: {datas}")
        logging.error(f"::예외 내용:: {e}")
        return []

def search_passages_to_milvus(partition_name:str, datas: list[int])->list[dict]:
    """
        주어진 ID 리스트를 기준으로 passages 컬렉션에서 원문 데이터를 조회한다.

        Parameters:
            partition_name (str): 조회할 파티션 이름
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
        from app.models.singleton import database_client # 순환 호출 문제로 인해 내부 import

        return database_client.milvus_client.get(
            collection_name="passages",
            partition_names=[partition_name],
            ids=datas,
            output_fields=[
                "passage"
            ],
        )
    except MilvusException as e:
        logging.error(f"::Error Exception(MilvusException):: passages에서 해당 ids로 entity 조회 실패")
        logging.error(f"::에러 발생 데이터:: {datas}")
        logging.error(f"::예외 내용:: {e}")
        raise EntityNotFound("passages에 해당 id로 엔티티가 존재하지 않습니다.")

def get_passages_id_from_triplets(similar_subjects: list[dict], similar_objects: list[dict], similar_relations: list[dict]) -> list[int]:
    """
        연관성 있는 주어, 서술어 문장 중 서로 연결된 정보들을 반환하는 함수이다.

        feat 명준: 코드 대부분이 지식 그래프 및 행렬에 대한 지식이 필요하여, 설명이 부족할 수 있다.

        주어진 주어, 목적어, 관계 정보를 바탕으로 그래프 상에서 연관된 관계들을 확장하고,
        이와 연결된 원문(passage)의 ID들을 반환한다. 주어-관계-목적어로 이루어진 인접 행렬을 생성하여
        2차 이웃까지 확장하며 관련 있는 관계를 탐색한다.

        Parameters:
            similar_subjects (list[dict]): Milvus에서 검색된 주어 관련 entity 정보 리스트.
            similar_objects (list[dict]): Milvus에서 검색된 목적어 관련 entity 정보 리스트.
            similar_relations (list[dict]): Milvus에서 검색된 관계(ner) 정보 리스트.

        Returns:
            list[int]: 연관된 관계로부터 도출된 원문 passage ID 목록.
    """
    # NOTE 1. 주어진 주어 및 목적어 리스트를 결합하여 하나의 개체 리스트로 구성
    triplets_with_similar_entities: list[dict] = []
    triplets_with_similar_entities.extend(similar_subjects)
    triplets_with_similar_entities.extend(similar_objects)

    # NOTE 2. 지식 그래프를 구성하기 위한 데이터 구조 초기화
    entities: list[str] = []  # 주어, 목적어 개체들
    relations: list[str] = []  # 관계(삼중항) 문자열: "주어 서술어 목적어"

    # 각 개체(주어/목적어) 인덱스에서 연결된 관계 인덱스들을 저장
    entityid_to_relationids = defaultdict(list)

    # Milvus에서 가져온 ID들과 내부 인덱스 매핑
    milvus_entity_id_to_index = dict()         # triplets_id → entity index
    milvus_relation_id_to_index = dict()       # triplets_id → relation index
    relationidx_to_passageids = defaultdict(set)  # relation index → passage id 집합

    # NOTE 3. 주어진 삼중항 데이터를 기반으로 그래프 구성
    for entity in triplets_with_similar_entities:
        subj = entity["entity"]["subject"]
        obj = entity["entity"]["object"]
        relation = entity["entity"]["relation"]
        triplets_id = entity["entity"]["triplets_id"]
        passages_id = entity["entity"].get("passages_id")  # 관계에 연결된 문장 ID

        # 엔티티 리스트에 존재하지 않으면 추가
        if subj not in entities:
            entities.append(subj)
        if obj not in entities:
            entities.append(obj)

        # 개체 인덱스를 구함
        subj_idx = entities.index(subj)
        obj_idx = entities.index(obj)

        # 관계 문자열 생성
        if relation not in relations:
            # 새로운 관계이면 등록 및 인덱스 저장
            relations.append(relation)
            relation_idx = len(relations) - 1

            # 개체-관계 간 연결성 저장
            entityid_to_relationids[subj_idx].append(relation_idx)
            entityid_to_relationids[obj_idx].append(relation_idx)

            # Milvus triplets ID ↔ relation index 매핑
            milvus_relation_id_to_index[triplets_id] = relation_idx
        else:
            relation_idx = relations.index(relation)

        # 관계 ID → passage ID 저장
        if passages_id is not None:
            if isinstance(passages_id, list):
                relationidx_to_passageids[relation_idx].update(passages_id)
            else:
                relationidx_to_passageids[relation_idx].add(passages_id)

        # triplets_id → 개체 인덱스 매핑
        milvus_entity_id_to_index[triplets_id] = subj_idx  # 또는 obj_idx

    # NOTE 4. 인접 행렬 구성 (개체-관계 간 연결성)
    entity_relation_adj = np.zeros((len(entities), len(relations)))
    for entity_id, entity in enumerate(entities):
        entity_relation_adj[entity_id, entityid_to_relationids[entity_id]] = 1
    entity_relation_adj = csr_matrix(entity_relation_adj)

    # NOTE 5. 행렬 곱셈을 통해 1-hop, 2-hop 인접 관계 계산
    entity_adj_1_degree = entity_relation_adj @ entity_relation_adj.T
    relation_adj_1_degree = entity_relation_adj.T @ entity_relation_adj

    # 확장 차수 설정 (2차 이웃까지)
    target_degree = 2

    entity_adj_target_degree = entity_adj_1_degree
    for _ in range(target_degree - 1):
        entity_adj_target_degree = entity_adj_target_degree * entity_adj_1_degree
    relation_adj_target_degree = relation_adj_1_degree
    for _ in range(target_degree - 1):
        relation_adj_target_degree = relation_adj_target_degree * relation_adj_1_degree

    # entity → relation 연결성 재구성
    entity_relation_adj_target_degree = entity_adj_target_degree @ entity_relation_adj

    # NOTE 6. 유사 관계 및 개체로부터 확장된 관계 인덱스 수집
    expanded_relations_from_relation = set()
    expanded_relations_from_entity = set()

    # 유사 관계에 대해 확장된 관계 수집
    filtered_hit_relation_ids = [
        milvus_relation_id_to_index[relation_res["triplets_id"]]
        for relation_res in similar_relations
        if relation_res["triplets_id"] in milvus_relation_id_to_index
    ]
    for hit_relation_idx in filtered_hit_relation_ids:
        expanded_relations_from_relation.update(
            relation_adj_target_degree[hit_relation_idx].nonzero()[1].tolist()
        )

    # 유사 개체에 대해 확장된 관계 수집
    filtered_hit_entity_ids = [
        milvus_entity_id_to_index[entity["entity"]["triplets_id"]]
        for entity in triplets_with_similar_entities
        if entity["entity"]["triplets_id"] in milvus_entity_id_to_index
    ]
    for hit_entity_idx in filtered_hit_entity_ids:
        expanded_relations_from_entity.update(
            entity_relation_adj_target_degree[hit_entity_idx].nonzero()[1].tolist()
        )

    # NOTE 7. 확장된 관계에 연결된 passage ID 추출
    relation_candidate_ids = list(
        expanded_relations_from_relation | expanded_relations_from_entity
    )

    passage_candidate_ids = set()
    for rid in relation_candidate_ids:
        passage_candidate_ids.update(relationidx_to_passageids.get(rid, set()))

    # 결과 반환
    return list(passage_candidate_ids)