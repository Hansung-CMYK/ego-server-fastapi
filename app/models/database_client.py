import os

from dotenv import load_dotenv
from pymilvus import MilvusClient

from parsed_sentence import ParsedSentence

# .env 환경 변수 추출
load_dotenv()
URI = os.getenv('URI')

class DatabaseClient:
    """
    Milvus를 이용하기 위한 Client이다.

    각종 함수에 활용된다.
    Attibutes:
        milvus_client:
    """
    TOKEN = "root:Milvus"

    def  __init__(self):
        self.milvus_client = MilvusClient(
            uri=URI,
            token=self.TOKEN
        )

    def search_all(self, collection_name: str, partition_names: list[str], field_name:str, output_fields: list[str]):
        """
        collection_name의 partition_name에 있는 모든 entity를 조회한다.

        id 값이 0 이상인 값을 조회하는 것을 조건으로 생성한다. (pymilvus 자체에는 전체 조회가 없다.)
        """
        return self.milvus_client.query(
            collection_name=collection_name,
            partition_names=partition_names,
            anns_field=field_name,
            filter=f"{field_name} >= 0",
            output_fields=output_fields
        )

    def insert_messages_into_milvus(
            self,
            splited_messages: list[str],
            partition_name: str
    ):
        """
        임베딩된 텍스트를 DB에 저장한다.

        :param splited_messages: 단일 문장으로 분리된 문장 리스트
        :param partition_name: 저장할 파티션 명
        """
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
            res = self.milvus_client.insert(
                collection_name="passages",
                partition_name=partition_name,
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

            self.milvus_client.insert(
                collection_name="triplets",
                partition_name=partition_name,
                data=triplet_datas
            )