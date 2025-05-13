from app.models.ner_model import ner_model
from app.models.embedding_model import embedding_model

class ParsedSentence:
    """
    speak 문장을 받아서 원문(passage)와 삼중항(triplets)로 저장하는 클래스이다.

    Attributes:
        passage (str): 사용자가 입력한 자연어 문장, 원문 텍스트.
        triplets (list[list[str]]): 추출된 삼중항 정보. 각 항목은 [주어, 서술어, 목적어]의 리스트 형태.
        relations (list[str]): 각 삼중항을 공백 기준으로 연결한 관계 표현 문자열 리스트.
    """
    def __init__(self, passage:str):
        """
        :param passage: 삼중항과 원문으로 저장할 문자열(문장)
        """
        # 순환 호출 문제로 인해 내부 import
        triplets = ner_model.extract_triplets(passage)
        self.passage = passage
        self.triplets = triplets["triplets"]
        self.relations = triplets["relations"]

    def embedding(self):
        """
        speak 정보를 임베딩하는 함수이다.

        Returns:
            dict:
                - embedded_passage (ndarray): 임베딩된 원문 벡터
                - embedded_triplets (list[list[ndarray]]): 각 삼중항(주어, 서술어, 목적어) 벡터들의 리스트
                - embedded_relations (list[ndarray]): 관계 문자열들의 임베딩 벡터 리스트
        """

        # NOTE 1. Speak 객체의 속성을 모두 임베딩 한다.
        embedded_triplets = [embedding_model.embed_documents(triplet) for triplet in self.triplets]
        embedded_relations = embedding_model.embed_documents(self.relations)
        embedded_passage = embedding_model.embed_documents([self.passage])[0]
        return {
            "embedded_triplets": embedded_triplets,
            "embedded_relations": embedded_relations,
            "embedded_passage": embedded_passage
        }