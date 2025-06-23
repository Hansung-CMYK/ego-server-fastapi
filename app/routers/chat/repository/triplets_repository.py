from numpy import ndarray

from config.database.milvus_database import MilvusDatabase

database = MilvusDatabase()


def insert_triplets(partition_name:str, data:dict|list[dict]):
    return database.insert(
        collection_name="triplets",
        partition_name=partition_name,
        data=data,
    )

def range_select_triplets(search_field:str, partition_names:list[str], output_fields:list[str], datas:ndarray|list[ndarray]):
    return database.range_select(
        collection_name="triplets",
        search_field=search_field,
        partition_names=partition_names,
        output_fields=output_fields,
        data=datas
    )

def select_all_triplets(partition_names: list[str], output_fields:list[str]):
    return database.select_all(
        collection_name="triplets",
        partition_names=partition_names,
        output_fields=output_fields,
    )

def select_triplets_to_ids(partition_names: list[str], ids: int|list[int], output_fields: list[str]):
    return database.select_passages_to_ids(
        collection_name="triplets",
        partition_names=partition_names,
        ids=ids,
        output_fields=output_fields
    )

def delete_triplets(partition_name: str, filter:str):
    return database.delete(
        collection_name="triplets",
        partition_name=partition_name,
        filter=filter
    )