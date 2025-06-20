from transformers import ElectraConfig, ElectraTokenizer

from app.internal.exception.error_code import ControlledException, ErrorCode
from config.emotion.model import ElectraForMultiLabelClassification
from config.emotion.multilabel_pipeline import MultiLabelPipeline


class EmotionClassifier:
    """
        사용 방법 
        
        emotionClassifier = EmotionClassifier()

        return emotionClassifier.predict(text)
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-goemotions")

        config = ElectraConfig.from_pretrained("monologg/koelectra-small-v3-goemotions")
        config.id2label = {
            0: "존경", 1: "감탄", 2: "호감", 3: "애정",
            4: "신뢰", 5: "친근감", 6: "연민", 7: "위로",
            8: "격려", 9: "감사", 10: "지원", 11: "희망",
            12: "분노", 13: "경멸", 14: "짜증", 15: "질투",
            16: "실망", 17: "불안", 18: "무관심", 19: "놀람",
            20: "호기심(긍정)", 21: "호기심(부정)", 22: "흥미", 23: "흥분",
            24: "슬픔(연민)", 25: "억울함", 26: "경계심", 27: "공감"
        }

        self.model = ElectraForMultiLabelClassification.from_pretrained(
            "monologg/koelectra-small-v3-goemotions", config=config
        )

        self.pipeline = MultiLabelPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            threshold=0.3
        )

    def predict(self, texts):
        raw = self.pipeline(texts)

        return [
            (label, score)
            for doc in raw
            for item in doc
            for label, score in zip(item["labels"], item["scores"])
        ]

    @staticmethod
    def mapper(relation:list[tuple]):
        """
            요약:
                각 감정의 BE.relationship ID를 매핑해서 반환하는 함수

            Parameters:
                relation(str): 도출된 감정
            """
        if relation == "존경": return 1
        elif relation == "감탄": return 2
        elif relation == "호감": return 3
        elif relation == "애정": return 4
        elif relation == "신뢰": return 5
        elif relation == "친근감": return 6
        elif relation == "연민":return 7
        elif relation == "위로": return 8
        elif relation == "격려": return 9
        elif relation == "감사": return 10
        elif relation == "지원": return 11
        elif relation == "희망": return 12
        elif relation == "분노": return 13
        elif relation == "경멸": return 14
        elif relation == "짜증": return 15
        elif relation == "질투": return 16
        elif relation == "실망": return 17
        elif relation == "불안": return 18
        elif relation == "무관심": return 19
        elif relation == "놀람": return 20
        elif relation == "호기심(긍정)": return 21
        elif relation == "호기심(부정)": return 22
        elif relation == "흥미": return 23
        elif relation == "흥분": return 24
        elif relation == "슬픔(연민)": return 25
        elif relation == "억울함": return 26
        elif relation == "경계심": return 27
        elif relation == "공감": return 28
        else:
            raise ControlledException(ErrorCode.INVALID_RELATIONSHIP)