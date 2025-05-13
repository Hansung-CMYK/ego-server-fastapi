import warnings
from textwrap import dedent

from langchain_core._api import LangChainDeprecationWarning
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 로깅 에러 문구 제거
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

class MainLlmModel:
    """
    Ollama를 통해 LLM 모델을 가져오는 클래스
    """

    # 세션 정보를 저장해두는 매핑 테이블
    __store : dict[str, ChatMessageHistory] = {}

    def __init__(self):
        MAIN_LLM_MODEL = "gemma3"

        model = ChatOllama(
            model=MAIN_LLM_MODEL,
            temperature=0.7
        )

        # 메인 모델 프롬프트 적용 + 랭체인 생성
        prompt = ChatPromptTemplate.from_messages(self.__MAIN_TEMPLATE)
        main_chain = prompt | model

        self.__prompt = RunnableWithMessageHistory(
            main_chain,
            self.__get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def get_store_keys(self):
        return self.__store.keys()

    def __get_session_history(self, session_id:str) -> BaseChatMessageHistory:
        """
        세션 아이디로 기존 대화 내역을 불러오는 함수

        :param session_id: 사용자의 세션 아이디
        :return: 최신 대화 내역이 기록된 객체
        """
        if session_id not in self.__store:
            self.__store[session_id] = ChatMessageHistory()
        return self.__store[session_id]

    def get_chain(self):
        return self.__prompt

    def get_human_messages_in_memory(self, session_id:str) -> list[str]:
        """
        langchain ConversationBufferMemory의 chat_memory로부터
        모든 메시지를 사람이 읽을 수 있는 문자열로 반환합니다.

        :return: list[str] 형태의 대화 내역 (예: ["나: ...", "친구: ..."])
        """
        memory = self.__get_session_history(session_id=session_id)

        message_list = [f"{'human' if msg.type == 'human' else 'ai' }: {msg.content}" for msg in memory.messages]

        return message_list

    __MAIN_TEMPLATE = [
        ("system", "/no_think\n"),
        ("system", dedent("""
                너는 나의 대화 상대야.
                - 답변은 'persona'를 기반으로 답해줘.
                - 답변은 2–4문장(30~80토큰) 안에서 간결-명확하게.
                - 제공된 'history'와 'related_story' 외 새로운 사실은 채택하지 말 것.
                - 'related_story'는 각 개체의 사건 + 의미 정보로 사건의 맥락을 인식할 것   
                - 필요하다면 인용부호 없이 자연스럽게 대화 기록을 재사용해.
            """).strip()),
        ("system", "{persona}"),
        MessagesPlaceholder(variable_name="history"),
        ("system", "{related_story}"),
        ("human", "{input}")
    ]

main_llm = MainLlmModel()