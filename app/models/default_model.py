from textwrap import dedent

from keybert import KeyBERT
from kiwipiepy import Kiwi
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

"""
요약:
    채팅을 생성하는 모델이다.

설명:
    MainLLM: 사용자에게 에고를 투영하여 알맞은 답변을 제공하는 모델이다.
"""
chat_model = ChatOllama( # MainLlm
    model="gemma3:12b",
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
{"name": function name, "parameters": dictionary of argument name and its value}

You SHOULD NOT include any other text in the response if you call a function
"""

"""
기타 모델
"""
kiwi = Kiwi() # 형태소 분석기(명사만 남기기 위함)
sentence_transformer = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
keyword_model = KeyBERT(sentence_transformer) # 한국어 SBERT