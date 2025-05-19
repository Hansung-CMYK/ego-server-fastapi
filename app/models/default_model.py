from keybert import KeyBERT
from kiwipiepy import Kiwi
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

"""
채팅을 생성하는 모델이다.

MainLLM: 사용자에게 에고를 투영하여 알맞은 답변을 제공하는 모델이다.
PreferenceModel: 관계 분석 
    - 파인 튜닝 예정
"""
chat_model = ChatOllama( # MainLlm
    model="gemma3:12b",
    temperature=0.7
)

# """
# 음성 채팅을 생성하는 모델이다.
# """
# speak_model = ChatOllama(
#     model="gemma3:4b",
#     temperature=0.7
# )

"""
특정 작업을 수행하는 LLM 모델에 사용하는 객체이다.

PersonaLlm: 사용자의 페르소나를 분석하고, 페르소나의 변화를 탐지하는 모델이다.
SplitLlm: 여러 의미를 가지고 있는 복합 문장을 단일의 사실과 의미로 분리하는 모델이다.
DailyCommentLLM: 일기 생성 시, 일기를 한 줄로 요약하는 모델이다.
TopicLlm: 일기의 주제를 추출하고, 그에 대한 이야기를 서술하는 모델이다. 
    - json으로 사용안하도록. 위험도가 있음
"""
task_model = ChatOllama(
    model="qwen3:8b",
    temperature=0.0,
    format="json"
)

kiwi = Kiwi() # 형태소 분석기(명사만 남기기 위함)
sentence_transformer = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
keyword_model = KeyBERT(sentence_transformer) # 한국어 SBERT