from json import JSONDecodeError
from textwrap import dedent
from langchain_core.prompts import ChatPromptTemplate
import json
import logging

from app.models.default_model import task_model


class PersonaLlm:
    """
    Ollama를 통해 LLM 모델을 가져오는 클래스

    대화 내역을 바탕으로 사용자의 페르소나를 재구성해주는 모델이다.
    """
    def __init__(self):
        """
        Ollama와 LangChain을 이용하여 LLM 모델을 생성
        """
        # 메인 모델 프롬프트 적용 + 랭체인 생성
        prompt = ChatPromptTemplate.from_messages(self.__PERSONA_TEMPLATE)
        self.__persona_chain = prompt | task_model

    def invoke(self, current_persona: str, session_history: str) -> dict:
        """
        사용자의 대화기록을 바탕으로 페르소나를 수정한다.

        :param current_persona: 현재 페르소나 json 정보
        :param session_history: 최근 대화 내역
        :return: json 정보를 가진 dict
        """
        answer = self.__persona_chain.invoke(
            {"session_history": session_history, "sample_json": self.__SAMPLE_JSON, "current_persona": [current_persona]}
        ).content
        clean_answer = self.__clean_json_string(answer)

        # dict로 자료형 변경
        try:
            return json.loads(clean_answer)
        except JSONDecodeError:
            logging.warning(f"::Error Exception(JSONDecodeError):: 변경할 persona가 없습니다. 변경사항을 반영하지 않습니다.")
            return {}

    @staticmethod
    def __clean_json_string(text: str) -> str:
        """
        LLM이 출력한 문자열에서 ```json 및 ``` 마커를 제거하고 공백을 정리한다.
        """
        text = text.strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text

    __PERSONA_TEMPLATE = [
        ("system", "/no_think\n"),
        ("system", dedent("""
            대화 기록을 보고 사용자의 Persona 변화를 JSON으로 정리

            [규칙]
            - CURRENT_PERSONA를 참고해 ‘변경, 추가’ 해야 할 필드만 무조건 JSON으로 출력
            - 새 키 만들지 말 것. 키 값은 SAMPLE_JSON와 허용 최상위 키 참고
            - 허용 최상위 키 (무조건 포함시킬 것): [`$set`, `$unset`]
            - 허용 상위 키:
                ['location(사용자 지역)', 'likes(좋아하는 것)', 'dislikes(싫어하는 것)', 
                'personality(성격)', personality(관심사), 'goal(목표)']
            - 추가 및 삭제 되어야 할 값은 정해진 키 값을 무조건 유지 - CURRENT_PERSONA 참고
            - 변경 및 추가할 정보는 `$set` 키에 저장 
            - 삭제할 정보는 `$unset`으로 저장

            [SAMPLE_JSON]
            {sample_json}

            [CURRENT_PERSONA]
            {current_persona}

            대화 기록:
            {session_history}
        """))
    ]

    __SAMPLE_JSON = """
            "$set": {
                "location": <str, 지역, 나라>,
                "likes": [<str, 좋아하는 것>, ...],
                "dislikes": [<str, 싫어하는 것>, ...]
                "personality": [<str, 성격>, ...],
                "goal": [str, 경제적 목표, ...]
            },
            "$unset": {...}
        """

# 싱글톤 생성
persona_llm = PersonaLlm()