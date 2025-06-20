import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

# 허깅페이스 로깅 레벨을 ERROR 이상으로 설정
hf_logging.set_verbosity_error()

class EmbeddingModel:
    """
    요약:
        주어진 문장을 벡터 임베딩을 하는 클래스

    설명:
        모델은 HuggingFace의 'dragonkue/snowflake-arctic-embed-l-v2.0-ko'를 사용하였다.

    Attributes:
        __tokenizer: 문장을 형태소 단위로 분리하기 위한 객체
        __model: 임베딩을 생성하기 위한 객체
        __device: 임베딩에 GPU를 사용하기 위한 객체
    """
    def __init__(self):
        EMBEDDINGS_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"

        self.__tokenizer = AutoTokenizer.from_pretrained(EMBEDDINGS_MODEL)
        self.__model = AutoModel.from_pretrained(EMBEDDINGS_MODEL, add_pooling_layer=False)
        self.__model.eval()
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU 사용 가능 시 연산을 GPU에서 하도록 변경
        self.__model.to(self.__device)

    def embedding(self, texts: list[str]) -> list[np.ndarray]:
        """
        요약:
            텍스트 리스트를 임베딩하는 함수

        Parameters:
            texts: 임베딩할 텍스트 리스트

        Returns:
            [embedded_text1, embedded_text2, ...]
        """
        # 토크나이징
        tokens = self.__tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=8192)
        tokens = {key: val.to(self.__device) for key, val in tokens.items()}

        # 임베딩 생성
        with torch.no_grad():
            outputs = self.__model(**tokens)[0][:, 0]  # CLS 토큰
            embeddings = torch.nn.functional.normalize(outputs, p=2, dim=1)

        # NumPy 배열로 반환
        return [embedding.cpu().numpy().astype(np.float16) for embedding in embeddings]

embedding_model = EmbeddingModel()