import re

from numpy import ndarray

from config.embedding.embedding_model import embedding_model
from config.ner.ner_service import split_to_triplets


class ParsedSentence:
    """
    요약:
        speak 문장을 받아서 원문(passage)와 삼중항(triplets)로 저장하는 클래스

    Attributes:
        passage(str): 사용자가 입력한 자연어 문장, 원문 텍스트.
        triplet(list[str]): 추출된 삼중항 정보. 각 항목은 [주어, 서술어, 목적어]
        relation(str): 각 삼중항을 공백 기준으로 연결한 관계 표현 문자열
    """
    def __init__(self, passage:str, single_sentence:str) -> None:
        """
        Parameters:
            passage(str): 삼중항과 원문으로 저장할 문자열(문장)
        """
        # 순환 호출 문제로 인해 내부 import
        triplets = split_to_triplets(single_sentence)
        self.passage = passage
        self.triplet:list = triplets["triplet"]
        self.relation:str = triplets["relation"]

    def element_embedding(self)->dict:
        """
        요약:
            speak 정보를 임베딩하는 함수이다.

        Returns:
            dict:
                - embedded_passage(ndarray): 임베딩된 원문 벡터
                - embedded_triplets(list[list[ndarray]]): 각 삼중항(주어, 서술어, 목적어) 벡터들의 리스트
                - embedded_relations(list[ndarray]): 관계 문자열들의 임베딩 벡터 리스트
        """

        # NOTE 1. Speak 객체의 속성을 모두 임베딩 한다.
        embedded_triplet:list[ndarray] = embedding_model.embedding(texts=self.triplet)

        embedded_relations:ndarray = embedding_model.embedding(texts=[self.relation])[0]
        return {
            "embedded_triplet": embedded_triplet,
            "embedded_relation": embedded_relations
        }

def split_sentence(text: str, delimiters: str = r"[.,!?;]") -> list[str]:
    """
    문장을 지정된 구분자로 분리하는 함수

    Parameters:
        text (str): 입력 문장
        delimiters (str): 구분자들을 포함한 정규표현식 패턴 (기본: . , ! ? ;)

    Returns:
        List[str]: 분리된 문장 리스트 (공백 제거됨)
    """
    # 구분자로 분리
    parts = re.split(delimiters, text)

    # 공백 제거 및 빈 문자열 제거
    return [part.strip() for part in parts if part.strip()]
