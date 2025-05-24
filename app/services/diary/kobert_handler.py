from collections import Counter, defaultdict
from app.models.emotion.emtion_classifier import EmotionClassifier

_labels = ['화남', '불안', '행복', '평범', '슬픔']

_mapping_28_to_5 = {
    "격앙된":    "화남",
    "까칠한":    "화남",
    "비판적인":  "화남",
    "혐오스러운": "화남",

    "불안한":    "불안",
    "긴장된":    "불안",
    "혼란스러운":"불안",
    "어색한":    "불안",
    "후회하는":  "불안",

    "즐거운":    "행복",
    "들뜬":      "행복",
    "기쁜":      "행복",
    "감사하는":  "행복",
    "사랑스러운":"행복",

    "우울한":    "슬픔",
    "비통한":    "슬픔",
}

def extract_emotions(
    contents: list[str],
    alpha: float = 0.5,
    top_k: int = 2
) -> list[str]:
    """
    Args:
        contents: 분석할 텍스트 목록
        alpha: 빈도 vs 확률 가중치 (0 ~ 1)
        top_k: 반환할 상위 감정 개수
    Returns:
        상위 k개 5개 레이블 리스트
    """

    classifier = EmotionClassifier()
    raw_pairs = classifier.predict(contents)

    preds, confs = [], []
    for label28, score in raw_pairs:
        label5 = _mapping_28_to_5.get(label28, "평범")
        preds.append(_labels.index(label5))
        confs.append(score)

    count = Counter(preds)
    sum_conf = defaultdict(float)
    for idx, c in zip(preds, confs):
        sum_conf[idx] += c

    total_count = sum(count.values()) or 1
    total_conf  = sum(sum_conf.values()) or 1.0
    norm_count = {i: count[i] / total_count for i in count}
    norm_conf  = {i: sum_conf[i] / total_conf  for i in sum_conf}

    score = {
        i: alpha * norm_count.get(i, 0.0)
           + (1 - alpha) * norm_conf.get(i, 0.0)
        for i in set(norm_count) | set(norm_conf)
    }

    top_indices = sorted(score, key=score.get, reverse=True)[:top_k]
    return [_labels[i] for i in top_indices]
