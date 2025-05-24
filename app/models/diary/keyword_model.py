from keybert import KeyBERT
from langchain.embeddings import sentence_transformer

from app.models.default_model import kiwi
from app.logger.logger import logger

class KeywordModel:
    """
    요약:
        문장에서 키워드(핵심 단어)를 추출하는 모델

    Attributes:
        __keyword_model(KeyBERT): 실질적으로 키워드를 추출하는 모델이다.
    """
    __keyword_model = KeyBERT(sentence_transformer)

    def get_keywords(self, chat_rooms:list[str], count:int=5)->list[str]:
        """
        요약:
            KeyBERT를 통해 벡터 임베딩을 통한 코사인 유사도로 키워드 추출

        Parameters:
            chat_rooms(list[str]): 키워드를 추출할 문장들의 모음
            count(int): 추출할 문장 개수
        """
        nouns_list = [] # 추출된 키워드를 담아 둘 객체

        # NOTE 1. 문장 하나로 연결
        sentences = "\n".join(story for story in chat_rooms)

        # NOTE 2. 문장에서 명사만 남겨둔다.
        for sentence in kiwi.analyze(sentences):
            nouns = [token.form for token in sentence[0] if token.tag.startswith('NN')]
            if nouns: nouns_list.extend(nouns)
        result_text = ' '.join(nouns_list)

        # NOTE 3. 키워드를 추출한다.
        # 상위 count개만큼 추출한다.
        original_value_keywords = self.__keyword_model.extract_keywords(result_text, keyphrase_ngram_range=(1, 1), top_n=count)
        keywords = [keyword[0] for keyword in original_value_keywords]

        # LOG. 시연용 로그2
        logger.info(msg=f"\nPOST: api/v1/diary [키워드]\n{keywords}\n")
        return keywords

keyword_model = KeywordModel()