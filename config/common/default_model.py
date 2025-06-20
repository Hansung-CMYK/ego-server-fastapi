import os
import threading

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()
MODEL_VERSION = os.environ.get('MODEL_VERSION')

"""
요약:
    채팅을 생성하는 모델이다.

설명:
    MainLLM: 사용자에게 에고를 투영하여 알맞은 답변을 제공하는 모델이다.
"""
chat_model = ChatOllama( # MainLlm
    model=MODEL_VERSION,
    temperature=0.7
)

"""
요약:
    특정 작업을 수행하는 LLM 모델에 사용하는 객체이다.

설명:
    PersonaLlm: 사용자의 페르소나를 분석하고, 페르소나의 변화를 탐지하는 모델이다.
    SplitLlm: 여러 의미를 가지고 있는 복합 문장을 단일의 사실과 의미로 분리하는 모델이다.
    DailyCommentLLM: 일기 생성 시, 일기를 한 줄로 요약하는 모델이다.
    TopicLlm: 일기의 주제를 추출하고, 그에 대한 이야기를 서술하는 모델이다. 
        - json으로 사용안하도록. 위험도가 있음
"""
task_model = chat_model

DEFAULT_TASK_LLM_TEMPLATE = """
You have access to functions. If you decide to invoke any of the function(s),
you MUST put it in the format of
{"result": dictionary of argument name and its value }

You SHOULD NOT include any other text in the response if you call a function
"""

"""
기타 모델
"""

import re


def clean_json_string(text: str) -> str:
    """
    요약:
        LLM이 출력한 문자열에서 ```json / ``` 마커와
        <think> ... </think> 블록을 제거하고 양쪽 공백을 정리한다.

    Parameters:
        text (str): 정제할 텍스트

    Returns:
        str: 정제된 문자열
    """
    # 양쪽 공백 제거
    text = text.strip()

    # 코드펜스 ```json ... ``` 제거
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    # <think> ... </think> 블록 제거 (여러 개 가능)
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

    return text.strip()


llm_sem = threading.Semaphore(1)