from keybert import KeyBERT
from kiwipiepy import Kiwi
from krwordrank.word import KRWordRank
from sentence_transformers import SentenceTransformer

class KeywordModel:
    __model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")  # 한국어 SBERT
    __keyword_model = KeyBERT(__model)
    __kiwi = Kiwi()

    def get_keywords_textranker(self, stories: list[list[str]], count: int = 5, beta=0.85, max_iter=10):
        # KRWordRank는 호출할 때마다 새로 생성
        wordrank_extractor = KRWordRank(
            min_count=1,  # 단어의 최소 출현 빈도수 (그래프 생성 시)
            max_length=100,  # 단어의 최대 길이
            verbose=True
        )

        # 텍스트 준비
        texts = [chat for story in stories for chat in story]
        full_text = " ".join(texts)

        # 명사 추출
        nouns = [
            token.form
            for analyzed in self.__kiwi.analyze(full_text)
            for token in analyzed[0]
            if token.tag.startswith("NN")
        ]
        result_text = " ".join(nouns)

        # 추출
        keywords, _, _ = wordrank_extractor.extract([result_text], beta=beta, max_iter=max_iter)

        return [word for word, _ in sorted(keywords.items(), key=lambda x: -x[1])[:count]]

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