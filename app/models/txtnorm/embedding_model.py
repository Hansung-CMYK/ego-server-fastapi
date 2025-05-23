import logging

import numpy as np
from transformers import AutoTokenizer, AutoModel

import torch

# 로깅 에러 문구 제거
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

class EmbeddingModel:
    """
    임베딩을 하는 클래스이다.

    모델은 HuggingFace의 'dragonkue/snowflake-arctic-embed-l-v2.0-ko'를 사용하였다.
    """
    def __init__(self):
        EMBEDDINGS_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"

        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDINGS_MODEL)
        self.model = AutoModel.from_pretrained(EMBEDDINGS_MODEL, add_pooling_layer=False)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU 사용 가능 시 연산을 GPU에서 하도록 변경
        self.model.to(self.device)

    def embeded_documents(self, texts: list[str]) -> list[np.ndarray]:
        """
        텍스트 리스트를 임베딩하는 함수
        Parameters:
            texts: 임베딩할 텍스트 리스트

        Return:
            임베딩된 문장의 벡터 정보
        """
        # 토크나이징
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=8192)
        tokens = {key: val.to(self.device) for key, val in tokens.items()}

        # 임베딩 생성
        with torch.no_grad():
            outputs = self.model(**tokens)[0][:, 0]  # CLS 토큰
            embeddings = torch.nn.functional.normalize(outputs, p=2, dim=1)

        # NumPy 배열로 반환
        return [embedding.cpu().numpy().astype(np.float16) for embedding in embeddings]

embedding_model = EmbeddingModel()