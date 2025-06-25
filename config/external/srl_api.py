import json
import os

import requests

from app.internal.exception.error_code import ControlledException, ErrorCode
from app.internal.logger.logger import logger

AI_OPEN_URL  = os.getenv('AI_OPEN_URL')
KEYS  = json.loads(os.getenv("AI_DATA_ACCESS_KEYS"))
KEYS_INDEX = 0

"""
AI_OPEN api

출처: https://aiopen.etri.re.kr/
"""
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