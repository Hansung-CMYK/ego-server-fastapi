from __future__ import annotations
import os
import numpy as np

from app.models.txtnorm.embedding_model import embedding_model

__TAGS = {
    "늘정주나": "정주행, 몰입, 콘텐츠, 연속시청, 시리즈완료, 스트리밍",
    "시간마술사": "지각, 시간조절, 변명, 알람연장, 일정조정, 지연",
    "존댓말자동완성": "예의, 어색함, 사회초년생, 경어체, 높임법, 존칭",
    "식사개근": "규칙적, 식사, 개근상, 삼시세끼, 도시락, 건강습관",
    "유교휴먼": "예의범절, 전통, 인사, 경례, 존경, 예절",
    "TMI장인": "정보과다, 수다, 과잉, 세부사항, 설명, 누설",
    "콘센트헌터": "충전기, 배터리, 생존본능, 전원탐색, 전력, 전원연결",
    "눈치n년차": "사회생활, 눈치, 숙련, 분위기파악, 상황판단, 기민함",
    "혼밥초년생": "혼자식사, 어색함, 단독식사, 좌석선택, 식사예절, 시선의식",
    "회식병가자": "회식기피, 피로, 회피, 음주회식, 사교부담, 휴식선호",
    "줄서기본능": "대기, 인기, 열풍, 대기행렬, 수요증가, 혼잡",
    "훈수스피커": "간섭, 참견, 조언, 지시, 간접통제, 의견제시",
    "애국열사": "국가사랑, 열성, 과도함, 국기, 애국심, 민족주의",
    "알코올엔진": "음주, 동력, 주당, 막걸리, 혼합주, 해장",
    "연예계CEO": "팬덤, 정보력, 기획자, 일정관리, 상품제작, 공연준비",
    "출퇴근금강불괴": "통근길, 체력, 인내, 혼잡열차, 장거리, 이동",
    "넷미인": "화면발, 필터, 반전, 사진보정, 촬영기술, 조명효과",
    "하트채굴자": "소셜미디어, 좋아요, 반응, 피드순회, 공유, 호응",
    "광기계발자": "열정, 코딩, 몰입, 심야작업, 결함수정, 버전관리",
    "상습괜찮러": "감정억제, 무덤덤, 인내, 침착, 수용, 자기억제",
    "배달VIP": "배달앱, 누적주문, 편의, 새벽배송, 주문기록, 택배",
    "수면채무자": "수면부족, 피로, 피곤, 불면, 각성제, 다크서클",
    "고민사색가": "심사숙고, 선택장애, 우유부단, 고찰, 비교, 숙려",
    "감정장대봉": "기분변화, 극단, 표현, 변동, 기복, 감정폭발",
    "결정권위임자": "선택포기, 타인위임, 귀찮음, 의존, 결정장애, 위탁",
    "MBTI목사": "성격유형, 과몰입, 전파, 유형분석, 검사, 유형홍보",
    "구매서기관": "장바구니, 후기, 정보수집, 최저가, 가격비교, 충동구매",
    "무도식여행가": "자유여행, 무계획, 즉흥, 현지체험, 방랑, 탐험",
    "인간명왕성": "거리감, 냉담, 고립, 관찰자, 소극성, 단절",
    "디지털인간": "비대면, 기술, 온라인, 가상공간, 원격근무, 인터넷",
    "엄친아판별기": "비교, 열등감, 평가, 경쟁심, 성과, 관찰",
    "K-약속러": "시간약속, 지각, 문화, 시간관리, 대기, 약속지연"
}


__SAVING_TAG_PATH = os.path.join(os.path.dirname(__file__), "keyword_index.npz")

def __build_tag(path: str = __SAVING_TAG_PATH) -> tuple[list[str], np.ndarray]:
    """
    태그 사전을 임베딩하고 임베딩 값을 .npz 파일에 저장한다.
    """
    keys, embeds = [], []
    for k, phrase in __TAGS.items():
        vec = embedding_model.embedding(texts=phrase)[0]  # (dim,)
        keys.append(k)
        embeds.append(vec.astype(np.float32))

    embeds = np.vstack(embeds)                              # (k, dim)
    np.savez_compressed(path, keys=np.asarray(keys), embeds=embeds)
    return keys, embeds

def load_index() -> tuple[list[str], np.ndarray]:
    """
    태그들의 key와 임베딩 값들을 반환한다.
    - 파일이 없으면 자동으로 생성 후 저장

    Returns:
        keys: list[str]
        embeds: np.ndarray[float32] (k, dim)
    """
    if not os.path.exists(__SAVING_TAG_PATH):
        return __build_tag(path=__SAVING_TAG_PATH)

    data = np.load(__SAVING_TAG_PATH, allow_pickle=True)
    keys   = data["keys"].tolist()        # ndarray → list
    embeds = data["embeds"].astype(np.float32)
    return keys, embeds