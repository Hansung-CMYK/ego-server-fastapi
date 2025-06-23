from numpy import ndarray

from config.database.milvus_database import MilvusDatabase

database = MilvusDatabase()

"""
DML
"""
def select_all_passages(partition_names: list[str], output_fields:list[str]):
    """
    원문(passage)을 전체 조회하는 함수

    Parameters:
        partition_names(list[str]): 조회할 파티션 묶음
        output_fields(list[str]): 반환받고 싶은 field 명
    """
    return database.select_all(
        collection_name="passages",
        partition_names=partition_names,
        output_fields=output_fields,
    )

def select_passages_to_ids(partition_names: list[str], output_fields: list[str], ids: int|list[int]):
    """
    원문(passage)을 id를 통해 조회하는 함수

    Parameters:
        partition_names(list[str]): 조회할 파티션 묶음
        output_fields(list[str]): 반환받고 싶은 field 명
        ids(int|list[int]): 조회할 passages_id 묶음
    """
    return database.select_passages_to_ids(
        collection_name="passages",
        partition_names=partition_names,
        output_fields=output_fields,
        ids=ids
    )

def range_select_passages(partition_names:list[str], output_fields:list[str], search_field:str, data:ndarray|list[ndarray]):
    """
    datas와 인접한 벡터를 가진 원문(passage)을 조회하는 함수

    Parameters:
        partition_names(list[str]): 조회할 파티션 묶음
        output_fields(list[str]): 반환받고 싶은 field 명
        search_field(str): 인접 벡터를 구할 벡터 필드
        data(ndarray|list[ndarray]): 인접 벡터를 구할 기준 벡터(임베딩 텍스트)
    """
    return database.range_select(
        collection_name="passages",
        partition_names=partition_names,
        output_fields=output_fields,
        search_field=search_field,
        data=data
    )

def insert_passages(partition_name:str, data:dict|list[dict]):
    """
    원문(passage)을 추가하는 함수

    Parameters:
        partition_name(str): 조회할 파티션 묶음
        data(ndarray|list[ndarray]): 인접 벡터를 구할 기준 벡터(임베딩 텍스트)
    """
    return database.insert(
        collection_name="passages",
        partition_name=partition_name,
        data=data,
    )

def delete_passages(partition_name: str, filter:str="id >= 0"):
    """
    원문(passage)을 삭제하는 함수

    Parameters:
        partition_name(str): 삭제할 파티션 묶음
        filter(str): 삭제할 원문의 조건
    """
    return database.delete(
        collection_name="passages",
        partition_name=partition_name,
        filter=filter
    )

"""
Partition
"""
def create_passages_partition(partition_name:str):
    """
    원문(passage) 콜렉션에 새로운 파티션을 추가하는 함수

    Parameters:
        partition_name(str): 추가할 파티션 명
    """
    return database.create_partition(
        collection_name="triplets",
        partition_name=partition_name
    )

def drop_passages_partition(partition_name:str):
    """
    원문(passage) 콜렉션에 새로운 파티션을 제거하는 함수

    Parameters:
        partition_name(str): 제거할 파티션 명
    """
    return database.drop_partition(
        collection_name="passages",
        partition_name=partition_name
    )

def has_passages_partition(partition_name:str):
    """
    원문(passage) 콜렉션에 이미 파티션이 존재하는지 확인하는 함수

    Parameters:
        partition_name(str): 확인할 파티션 명
    """
    return database.has_partition(
        collection_name="passages",
        partition_name=partition_name
    )