from collections import Counter, defaultdict
from app.models.emotion.emotion_classifier import EmotionClassifier

_labels = ['화남', '불안', '행복', '평범', '슬픔']

_mapping_28_to_5 = {
    "존경": "행복", "감탄": "행복", "호감": "행복", "애정": "행복",
    "신뢰": "행복", "친근감": "행복", "위로": "행복", "격려": "행복",
    "감사": "행복", "지원": "행복", "희망": "행복", "공감": "행복",
    "놀람": "행복", "호기심(긍정)": "행복", "흥미": "행복", "흥분": "행복",
    "분노": "화남", "경멸": "화남", "짜증": "화남", "질투": "화남", "억울함": "화남",
    "불안": "불안", "경계심": "불안",
    "실망": "슬픔", "연민": "슬픔", "슬픔(연민)": "슬픔",
    "무관심": "평범", "호기심(부정)": "평범"
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
    return [_labels[i] for i in top_indices] if len([_labels[i] for i in top_indices]) >= top_k else ["평범"]
