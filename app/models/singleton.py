from app.models.database_client import DatabaseClient
from app.models.embedding_model import EmbeddingModel
from app.models.main_llm_model import MainLlmModel
from app.models.ner_model import NerModel
from app.models.split_llm_model import SplitLlmModel

"""
자주 사용되고, 단일 인스턴스만 필요한 객체들을 묶어 객체로 선언하는 파일이다.
"""
main_llm = MainLlmModel()
parsing_llm = SplitLlmModel()
database_client = DatabaseClient()
embedding_model = EmbeddingModel()
ner_model = NerModel()