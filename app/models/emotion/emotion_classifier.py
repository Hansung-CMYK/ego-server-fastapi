from transformers import ElectraTokenizer, ElectraConfig
from app.models.emotion.model import ElectraForMultiLabelClassification
from app.models.emotion.multilabel_pipeline import MultiLabelPipeline

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
