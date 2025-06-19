import json
import re, requests, os
from dotenv import load_dotenv

from app.internal.exception.error_code import ControlledException, ErrorCode
from app.internal.logger.logger import logger

load_dotenv()

AI_OPEN_URL  = os.getenv('AI_OPEN_URL')
KEYS  = json.loads(os.getenv("AI_DATA_ACCESS_KEYS"))
KEYS_INDEX = 0

def get_srl(text:str):
    """
    요약:
        의미역 인식(Semantic Role Labeling) API 호출

    Parameters:
        text(str): 의미역 인식을 요청할 단일 문장
    """
    global KEYS_INDEX
    body = {"argument":{"analysis_code":"srl", "text":text}}

    # AI API-DATA 언어 분석 기술 OPEN API 요청
    while True:
        try:
            return requests.post(
                url=AI_OPEN_URL,
                headers={"Authorization": KEYS[KEYS_INDEX], "Content-Type": "application/json"},
                json=body
            ).json()
        except json.JSONDecodeError:
            logger.info(msg=f"\n\nPOST: AI API DATA [구문 분석 json 파싱 실패]\n{text}\n")
            raise ControlledException(ErrorCode.FAILURE_JSON_PARSING)
        except Exception:
            KEYS_INDEX += 1
            if KEYS_INDEX == len(KEYS):
                raise ControlledException(ErrorCode.OUT_OF_BOUND_KEYS)
            continue

def expand(pharse_id:int, phrases:dict)->str:
    """
    요약:
        수식어 구를 재귀적으로 탐색해 확장하는 함수

    Parameters:
        pharse_id(int): 추가 필요
        phrases(dict): 확장 필요
    """
    phrase, text = phrases[pharse_id], phrases[pharse_id]["text"]
    for sub in phrase.get("sub_phrase", []):
        text = re.sub(rf"P#{sub}@\w+", expand(sub, phrases), text)
    text = re.sub(r"P#\d+@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def word2phrase(word_id:int, phrases:dict)-> int | None:
    """
    SRL argument는 word_id 기준이므로 phrase_dependency 상에서
    해당 단어를 감싸는 최소 phrase를 찾아야 한다.
    """
    cand = [phrase for phrase in phrases.values() if phrase["begin"] <= word_id <= phrase["end"]]
    return min(cand, key=lambda p:p["end"]-p["begin"])["id"] if cand else None

# 논항으로 추출할 우선순위: [참고:언어 분석 API 레퍼런스, 의미역 인식 태그셋](https://aiopen.etri.re.kr/guide/WiseNLU)
ARG_ORDER = [
    "ARG0","ARG1","ARG2","ARG3","ARG4",
    "ARGM-TMP","ARGM-LOC","ARGM-DIR","ARGM-MNR","ARGM-PRP",
    "ARGM-CAU","ARGM-EXT","ARGM-CND","ARGM-PRD","ARGM-DIS",
    "ARGM-ADV","ARGM-NEG"
]

def split_to_triplets(single_sentence:str)->dict:
    """
    요약:
        문장을 우선순위에 맞게 삼중항으로 삼등분한다.

    설명:
        삼중항으로 문장을 분리한다.
        만약 주어-목적어로 삼중항을 분리하지 못할 시, ARG_ORDER에 맞춰 공간을 할당하게 된다.

        파라미터 정보
        - INDEX 1: 첫번째 적격 논항 (ARG_ORDER 순서대로 탐색)
        - INDEX 2: 두번째 적격 논항
        - INDEX 3: 원문 (strip)
        - 예외: 논항 부족 시 빈 문자열("") 채운다.

    Parameters:
        single_sentence(str): 분리할 삼중항 단일 문장
    """
    nlu  = get_srl(single_sentence) # Natural Language Understanding
    sentence = nlu["return_object"]["sentence"][0]
    phrases = {p["id"]:p for p in sentence["phrase_dependency"]}

    # 예외처리: API에 SRL 프레임이 없으면 빈 리스트 반환
    if not sentence["SRL"]:
        return {
        "triplet": ["", "", single_sentence.strip()],
        "relation": sentence["text"]
    }

    frame = sentence["SRL"][0]
    arguments   = {argument["type"]:argument for argument in frame["argument"]}

    picked = []
    for tag in ARG_ORDER:
        if tag in arguments:
            pharse_id = word2phrase(arguments[tag]["word_id"], phrases)
            picked.append(expand(pharse_id, phrases))
        if len(picked) == 2:
            break
    while len(picked) < 2:
        picked.append("")

    return {
        "triplet": [picked[0], picked[1], single_sentence.strip()],
        "relation": sentence["text"]
    }