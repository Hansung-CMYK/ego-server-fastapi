from keybert import KeyBERT
from kiwipiepy import Kiwi

from app.models.default_model import sentence_transformer


class KeywordModel:
    __keyword_model = KeyBERT(sentence_transformer)
    __kiwi = Kiwi()

    def get_keywords(self, stories:list[list[str]], count:int=5):
        """
        KeyBERT를 통해 벡터 임베딩을 통한 코사인 유사도로 키워드 추출
        """
        nouns_list = []

        texts = [chat for story in stories for chat in story]
        sentences = " ".join(text for text in texts)

        for sentence in self.__kiwi.analyze(sentences):
            nouns = [token.form for token in sentence[0] if token.tag.startswith('NN')]
            if nouns: nouns_list.extend(nouns)
        result_text = ' '.join(nouns_list)

        original_value_keywords = self.__keyword_model.extract_keywords(result_text, keyphrase_ngram_range=(1, 1), top_n=count)
        keywords = [keyword[0] for keyword in original_value_keywords]
        return keywords

keyword_model = KeywordModel()