import re
from collections import Counter, defaultdict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification

_tokenizer = AutoTokenizer.from_pretrained('monologg/kobert', trust_remote_code=True)
_model     = BertForSequenceClassification.from_pretrained(
    'jeonghyeon97/koBERT-Senti5',
    trust_remote_code=True
)

_labels = ['화남','불안','행복','평범','슬픔']


def extract_emotions(topics: list[dict], alpha: float = 0.5, top_k: int = 2):
    """
    Args:
        txtnorm (str): 분류 대상 텍스트
        alpha (float): 빈도 vs 확률 가중치 (0~1, default=0.5)
        top_k (int): 반환할 상위 감정 개수 (default=2)

    Returns:
        List[str]: 상위 k개 감정 레이블
    """
    contents = [t['content'] for t in topics]

    # 2) 토큰화 & 예측
    inputs = _tokenizer(contents, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        logits = _model(**inputs).logits  
    probs = F.softmax(logits, dim=-1)    

    # 3) 문장별 예측 및 confidence
    preds = probs.argmax(dim=-1).tolist()
    confs = probs.max(dim=-1).values.tolist()

    # 4) 빈도 및 확률 합산 계산
    count = Counter(preds)
    sum_conf = defaultdict(float)
    for idx, c in zip(preds, confs):
        sum_conf[idx] += c

    # 5) 정규화
    total_count = sum(count.values())
    total_conf  = sum(sum_conf.values())
    norm_count = {i: count[i] / total_count for i in count}
    norm_conf  = {i: sum_conf[i] / total_conf  for i in sum_conf}

    # 6) 가중 합산 스코어
    score = {}
    all_idx = set(norm_count) | set(norm_conf)
    for i in all_idx:
        f = norm_count.get(i, 0.0)
        p = norm_conf .get(i, 0.0)
        score[i] = alpha * f + (1 - alpha) * p

    # 7) 상위 k개 반환
    top_indices = sorted(score, key=lambda i: score[i], reverse=True)[:top_k]
    return [_labels[i] for i in top_indices]
