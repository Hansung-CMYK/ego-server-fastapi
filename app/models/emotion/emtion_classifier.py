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
        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-goemotions")

        config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-goemotions")
        config.id2label = {
            0:"존경스러운", 1:"즐거운", 2:"격앙된", 3:"까칠한", 4:"수용적인",
            5:"배려 깊은", 6:  "혼란스러운", 7:  "호기심 많은", 8:  "매혹적인", 9:  "실망스러운",
            10: "비판적인", 11: "혐오스러운", 12: "어색한", 13: "들뜬", 14: "불안한",
            15: "감사하는", 16: "비통한", 17: "기쁜", 18: "사랑스러운", 19: "긴장된",
            20: "긍정적인", 21: "자부심 있는",22: "깨달은", 23: "안도하는", 24: "후회하는",
            25: "우울한", 26: "놀란", 27: "무난한"
        }

        self.model = ElectraForMultiLabelClassification.from_pretrained(
            "monologg/koelectra-base-v3-goemotions", config=config
        )

        self.pipeline = MultiLabelPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            threshold=0.3
        )

    def predict(self, texts):
        return self.pipeline(texts)
