"""
keyword_index.py
- 키워드 사전 → 벡터로 임베딩 → keyword_index.npz 에 저장
- 이미 저장돼 있으면 재임베딩 없이 즉시 로드
"""
from __future__ import annotations
import os
import numpy as np
from app.models.normalization.embedding_model import embedding_model

# ────────────────────────────────────────────────────────
_KEYWORDS={
    "늘정주나": "정주행 몰입 콘텐츠",
    "시간마술사": "지각 시간조절 변명",
    "존댓말자동완성": "예의 어색함 사회초년생",
    "식사개근": "규칙적 식사 개근상",
    "유교휴먼": "예의범절 전통 인사",
    "TMI장인": "정보과다 수다 오지랖",
    "콘센트헌터": "충전기 배터리 생존본능",
    "눈치n년차": "사회생활 눈치 경력자",
    "혼밥초년생": "혼자 식사 어색함",
    "회식병가자": "회식기피 피로 회피",
    "줄서기본능": "대기 인기 무조건",
    "훈수스피커": "간섭 참견 조언",
    "애국열사": "국가사랑 극성 감정과잉",
    "알코올엔진": "술 동력 주당",
    "연예계CEO": "팬덤 정보력 기획자",
    "출퇴근금강불괴": "출근길 체력 철인",
    "넷미인": "화면발 필터 반전",
    "하트채굴자": "SNS 좋아요 눈치",
    "광기계발자": "열정 코딩 몰입",
    "상습괜찮러": "참기 괜찮아요 감정억제",
    "배달VIP": "배달앱 누적주문 귀차니즘",
    "수면채무자": "잠부족 피로 빚",
    "고민사색가": "깊은생각 선택장애 우유부단",
    "감정장대봉": "감정기복 극단 표현",
    "결정권위임자": "선택포기 타인위임 귀찮음",
    "MBTI목사": "성격 과몰입 유형전도",
    "구매서기관": "장바구니 후기 정보력",
    "무도식여행가": "자유여행 무계획 즉흥",
    "인간명왕성": "거리감 냉담 미지",
    "디지털인간": "비대면 기술 온라인",
    "엄친아판별기": "비교 열등감 관찰자",
    "K-약속러": "시간약속 지각 한국인습관"
}

_SAVE_PATH = os.path.join(os.path.dirname(__file__), "keyword_index.npz")


def _build_and_save(path: str = _SAVE_PATH) -> tuple[list[str], np.ndarray]:
    """키워드 사전을 임베딩하고 .npz 파일에 저장한다."""
    keys, embeds = [], []
    for k, phrase in _KEYWORDS.items():
        vec = embedding_model.embeded_documents(phrase)[0]  # (dim,)
        keys.append(k)
        embeds.append(vec.astype(np.float32))

    embeds = np.vstack(embeds)                              # (k, dim)
    np.savez_compressed(path, keys=np.asarray(keys), embeds=embeds)
    return keys, embeds


def load_index() -> tuple[list[str], np.ndarray]:
    """
    (keys, embeds)를 반환한다.
    - 파일이 없으면 자동으로 생성 후 저장
    - keys: list[str]          embeds: np.ndarray[float32] (k, dim)
    """
    if not os.path.exists(_SAVE_PATH):
        return _build_and_save(_SAVE_PATH)

    data = np.load(_SAVE_PATH, allow_pickle=True)
    keys   = data["keys"].tolist()        # ndarray → list
    embeds = data["embeds"].astype(np.float32)
    return keys, embeds

if __name__ == "__main__":
    _build_and_save()