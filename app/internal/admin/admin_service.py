from pymilvus import DataType

from app.internal.admin.dto.admin_request import ADMIN_ID, ADMIN_PASSWORD
from app.routers.chat.repository import (passages_repository,
                                         triplets_repository)
from app.routers.chat.service.parsed_sentence import ParsedSentence
from app.routers.persona import persona_service
from app.routers.tone import tone_service
from config.database.milvus_database import MilvusDatabase
from config.database.postgres_database import PostgresDatabase
from config.external import hub_api


def check_authorization(admin_id:str, admin_password:str)->bool:
    return admin_id == ADMIN_ID and admin_password == ADMIN_PASSWORD

def check_correct_user(user_id:str, ego_id:str)->bool:
    """
    user_id와 ego_id가 같은 사용자의 것인지 확인하는 함수
    """
    return hub_api.get_ego(user_id=user_id) == int(ego_id)

"""
MILVUS INSERT DATA
"""
database = MilvusDatabase()

def init_passages():
    """
    passages 컬렉션을 생성하는 함수이다.
    """
    # NOTE 1. 이미 콜렉션이 존재하는지 확인한다.
    if database.has_collection(collection_name="passages"): # 만약에 이미 컬렉션이 존재한다면,
        database.drop_collection(collection_name="passages") # 기존 컬렉션을 삭제한다.

    # NOTE 2. passages 컬렉션 생성에 활용할 스키마 생성
    passages_schema = database.create_schema() # AUTO_INCREMENT
    passages_schema.add_field(field_name='passages_id', datatype=DataType.INT64, is_primary=True) # 아이디(기본키)
    passages_schema.add_field(field_name='passage', datatype=DataType.VARCHAR, max_length=255) # passage
    passages_schema.add_field(field_name='embedded_passage',datatype=DataType.FLOAT16_VECTOR,dim=1024) # embedded_passage
    passages_schema.add_field(field_name="is_fix",datatype=DataType.BOOL) # 삭제 가능 여부

    passages_index_params = database.prepare_index_params()
    passages_index_params.add_index(field_name='passages_id', index_name='passages_id_index', index_type='AUTOINDEX')
    passages_index_params.add_index(field_name='embedded_passage', index_name='embedded_passage_index',
                                    index_type='AUTOINDEX')

    # NOTE 3. passages 컬렉션을 생성한다.
    database.create_collection(collection_name="passages", schema=passages_schema, index_params=passages_index_params)

def init_triplets():
    """
    triplets 컬렉션을 생성하는 함수이다.
    """
    # NOTE 1. 이미 콜렉션이 존재하는지 확인한다.
    if database.has_collection(collection_name="triplets"):  # 만약에 이미 컬렉션이 존재한다면,
        database.drop_collection(collection_name="triplets")  # 기존 컬렉션을 삭제한다.

    # NOTE 2. triplets 컬렉션 생성에 활용할 스키마 생성
    triplets_schema = database.create_schema()  # AUTO_INCREMENT
    triplets_schema.add_field(field_name='triplets_id', datatype=DataType.INT64, is_primary=True)  # 아이디(기본키)
    triplets_schema.add_field(field_name='passages_id', datatype=DataType.INT64)  # passage_id(외래키)
    triplets_schema.add_field(field_name='subject', datatype=DataType.VARCHAR, max_length=255)  # 주어
    triplets_schema.add_field(field_name='object', datatype=DataType.VARCHAR, max_length=255)  # 목적어
    triplets_schema.add_field(field_name='relation', datatype=DataType.VARCHAR,
                              max_length=512)  # 관계 (문장 길이를 고려해 주어, 목적어 보다 증축)
    triplets_schema.add_field(field_name='embedded_subject', datatype=DataType.FLOAT16_VECTOR, dim=1024)  # 임베딩 된 주어
    triplets_schema.add_field(field_name='embedded_object', datatype=DataType.FLOAT16_VECTOR, dim=1024)  # 임베딩 된 목적어
    triplets_schema.add_field(field_name='embedded_relation', datatype=DataType.FLOAT16_VECTOR, dim=1024)  # 임베딩 된 관계
    triplets_schema.add_field(field_name="is_fix", datatype=DataType.BOOL)  # 삭제 가능 여부

    triplets_index_params = database.prepare_index_params()
    triplets_index_params.add_index(field_name='triplets_id', index_name='triplets_id_index', index_type='AUTOINDEX')
    triplets_index_params.add_index(field_name='passages_id', index_name='passages_id_index', index_type='AUTOINDEX')
    triplets_index_params.add_index(field_name='embedded_subject', index_name='embedded_subject_index',
                                    index_type='AUTOINDEX')
    triplets_index_params.add_index(field_name='embedded_object', index_name='embedded_object_index',
                                    index_type='AUTOINDEX')
    triplets_index_params.add_index(field_name='embedded_relation', index_name='embedded_relation_index',
                                    index_type='AUTOINDEX')

    # NOTE 3. triplets 컬렉션을 생성한다.
    database.create_collection(collection_name="triplets", schema=triplets_schema, index_params=triplets_index_params)

def load_partition(collection_name:str, partition_names: str|list[str]):
    database.load_partitions(collection_name=collection_name, partition_names=partition_names)

def insert_sample_messages(
        single_sentences: list[str],
        ego_id: str
):
    """
    임베딩된 텍스트를 DB에 저장한다.

    :param single_sentences: 단일 문장으로 분리된 문장 리스트
    :param ego_id: 저장할 파티션 명
    """
    # 문장을 삼중항으로 Parsing한다.
    parsed_sentences = [ParsedSentence(passage=single_sentece, single_sentence=single_sentece) for single_sentece in single_sentences]

    # NOTE 1. Passages에 값을 저장한다.
    for parsed_sentence in parsed_sentences:
        embedded_sentence = parsed_sentence.element_embedding()
        passage_data = {
            "passage": parsed_sentence.passage,
            "embedded_passage": embedded_sentence["embedded_relation"],
            "is_fix": True
        }

        # 실제 DB에 저장
        res = passages_repository.insert_passages(
            partition_name=ego_id,
            data=passage_data
        )
        passages_ids = res["ids"][0]  # res는 저장된 원문 ids 값

        # NOTE 2. triplets에 값을 저장한다.
        triplets_repository.insert_triplets(
            partition_name=ego_id,
            data={
                "passages_id": passages_ids,
                "subject": parsed_sentence.triplet[0],
                "object": parsed_sentence.triplet[1],
                "relation": parsed_sentence.relation,
                "embedded_subject": embedded_sentence["embedded_triplet"][0],
                "embedded_object": embedded_sentence["embedded_triplet"][1],
                "embedded_relation": embedded_sentence["embedded_relation"],
                "is_fix": True
            }
        )

"""
PERSONA, TONE TABLE INSERT DATA
"""
def init_persona():
    if persona_service.has_persona():
        persona_service.drop_persona()
    persona_service.create_persona()

def init_tone():
    if tone_service.has_tone():
        tone_service.drop_tone()
    tone_service.create_tone()