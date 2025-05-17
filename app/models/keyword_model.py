from keybert import KeyBERT
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer

class KeywordModel:
    __model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")  # 한국어 SBERT
    __keyword_model = KeyBERT(__model)
    __kiwi = Kiwi()

    def get_keywords_vector_embedding(self, stories:list[list[str]], count:int=5):
        """
        KeyBERT를 통해 벡터 임베딩을 통한 코사인 유사도로 키워드 추출
        """
        nouns_list = []

        texts = [chat for story in stories for chat in story]
        sentences = " ".join(text for text in texts)

        for sentence in self.__kiwi.analyze(sentences):
            nouns = [token.form for token in sentence[0] if token.tag.startswith('NN')]
            if nouns:
                nouns_list.extend(nouns)
        result_text = ' '.join(nouns_list)

        # TODO: 아키텍처(architecture)&환경(enviroment)에 따라 에러가 발생할 수 있다.
        """
        sklearn/utils/extmath.py:203: RuntimeWarning: divide by zero encountered in matmul
          ret = a @ b
        sklearn/utils/extmath.py:203: RuntimeWarning: overflow encountered in matmul
          ret = a @ b
        sklearn/utils/extmath.py:203: RuntimeWarning: invalid value encountered in matmul
          ret = a @ b
        참고: https://stackoverflow.com/questions/76527556/what-is-the-cause-of-runtimewarning-invalid-value-encountered-in-matmul-ret
        """
        original_value_keywords = self.__keyword_model.extract_keywords(result_text, keyphrase_ngram_range=(1, 1), top_n=count)
        keywords = [keyword[0] for keyword in original_value_keywords]
        return keywords

keyword_model = KeywordModel()