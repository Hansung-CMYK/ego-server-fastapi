import json
import os
import re
import threading
from abc import ABC, abstractmethod
from textwrap import dedent
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.internal.exception.error_code import ControlledException, ErrorCode
from app.internal.logger.logger import logger

load_dotenv()

MODEL_VERSION = os.environ.get('MODEL_VERSION')

"""
요약:
    채팅을 생성하는 모델이다.

설명:
    MainLLM: 사용자에게 에고를 투영하여 알맞은 답변을 제공하는 모델이다.
"""
chat_model = ChatOllama(
    model=MODEL_VERSION,
    temperature=0.7
)


class CommonLLM(ABC):
    """
    요청에 대한 JSON 값을 반환하는 LLM 추상 클래스

    JSON을 활용하는 LLM 구현 시, 꼭 다음 함수를 super()를 통해 이용해주세요.

    Attributes:
        _instances: 자식 클래스들의 싱글턴 인스턴스입니다.
            - _template(list): 각 prompt를 연결할 객체이다. 추가 TEMPLATE는 이 객체에 .append() 할 것
        _lock: 싱글턴을 구현하기 위한 동기화 Flag 객체입니다.

        _common_model(ChatOllama): CommonModel이 사용하는 ollama 모델
        _semaphore(Semaphore): Ollama 프로세스 수를 고정하기 위한 세마포

        _COMMON_COMMAND_TEMPLATE(tuple): LLM System Prompt - 제어 메타 태그
            - /json: 반환 값을 json 문자열로 반환한다.
            - /no_think: Qwen3의 경우 chain_of_thought를 결과를 출력하지 않도록 함
        _COMMON_RESPONSE_TEMPLATE(tuple): LLM System Prompt - 반환값을 JSON으로 고정하기 위한 명령어
    """
    _instances: dict[type, 'CommonLLM'] = {}
    _lock = threading.Lock()

    # _common_model = ChatOllama(
    #     model=MODEL_VERSION,
    #     temperature=0.0
    # )
    _common_model = chat_model
    _semaphore = threading.Semaphore(1)

    _COMMON_COMMAND_TEMPLATE = ("system", dedent("""
        /json
        /no_think
        
        You have access to functions. If you decide to invoke any of the function(s),
        you MUST put it in the format of
        {{"result": dictionary of argument name and its value }}

        You SHOULD NOT include any other text in the response if you call a function
        """))

    def __new__(cls, *args, **kwargs):
        """
        싱글턴 구현을 위한 함수입니다.

        인스턴스를 호출할 땐 CommonLLM() 혹은 상속 객체를 호출해주세요.
        """
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__new__(cls)
                instance._template = [
                    cls._COMMON_COMMAND_TEMPLATE,
                    # 상속받은 자식 클래스에서 추가적으로 Template를 추가할 수 있도록 TemplatePattern을 적용
                    *instance._add_template()
                ]
                instance._chain = (ChatPromptTemplate.from_messages(instance._template) | cls._common_model)
                cls._instances[cls] = instance
        return cls._instances[cls]


    @abstractmethod
    def _add_template(self)->list[tuple]:
        """
        추가할 프롬프트를 추가하는 함수

        추가 확장을 위해 TemplatePattern 활용
        """
        pass

    def invoke(self, parameter:dict)->Any:
        """
        LLM의 응답을 받는 함수입니다.

        Returns:
            parameter(dict): Template에 들어가야 할 인자 값

        Raises:
            FAILURE_JSON_PARSING: JSON Decoding 실패 시, 빈 딕셔너리 반환
        """
        with self._semaphore:
            answer:str = self._chain.invoke(parameter).content
        clean_answer:str = self.clean_json_string(text=answer)

        # LOG. 사연용 로그
        logger.info(msg=f"\n\n[{self.__class__.__name__}] invoke()\n{clean_answer}\n")

        # 반환된 문자열 dict로 변환
        try:
            return json.loads(clean_answer)["result"]
        except json.JSONDecodeError:
            raise ControlledException(ErrorCode.FAILURE_JSON_PARSING)
        except KeyError:
            raise ControlledException(ErrorCode.INVALID_DATA_TYPE)

    @staticmethod
    def clean_json_string(text: str) -> str:
        """
        요약:
            LLM이 출력한 문자열에서 \`\`\`json, \`\`\` 마커와
            <think> ... </think> 블록을 제거하고 양쪽 공백을 정리한다.

        Parameters:
            text(str): 정제할 텍스트

        Returns:
            Filtered Text
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