from database_client import DatabaseClient
from embedding_model import EmbeddingModel
from main_llm_model import MainLlmModel
from ner_model import NerModel
from pasing_llm_model import ParsingLlmModel

"""
자주 사용되고, 단일 인스턴스만 필요한 객체들을 묶어 객체로 선언하는 파일이다.
"""
main_llm = MainLlmModel()
parsing_llm = ParsingLlmModel()
database_client = DatabaseClient()
embedding_model = EmbeddingModel()
ner_model = NerModel()