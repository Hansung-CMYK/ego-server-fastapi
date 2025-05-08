import datetime
import os
from textwrap import dedent

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from exception.incorrect_answer import IncorrectAnswer

# .env 환경 변수 추출
load_dotenv()
TRIPLET_LLM_MODEL = os.getenv('TRIPLET_LLM_MODEL')

"""
Ollama를 통해 LLM 모델을 가져오는 클래스
"""
class ParsingLlmModel:
    def __init__(self):
        model = ChatOllama(
            model=TRIPLET_LLM_MODEL,
            temperature=0.0,
            format="json"
        )

        # 문장 분리 프롬프트 적용 + 랭체인 생성
        split_prompt = ChatPromptTemplate.from_messages(self.__SPLIT_TEMPLATE)
        self.__split_chain = split_prompt | model

        # 삼중항 모델 프롬프트 적용 + 랭체인 생성
        parsing_prompt = ChatPromptTemplate.from_messages(self.__SPEAK_TEMPLATE)
        self.__parsing_chain = parsing_prompt | model


    def parsing_invoke(self, passage: str) -> dict:
        """
        전달받은 문장을 삼중항으로 분리한다.
        TODO: 현재 사용하지 않음

        :param passage: 삼중항으로 분리할 문장
        :return:
        """
        import json
        # NOTE 1. 원문을 삼중항으로 분리한다.
        speak_dict_string = self.__parsing_chain.invoke({"input": passage, "example": ParsingLlmModel.SPEAK_TEMPLATE_EXAMPLE}).content.strip()

        # NOTE 2. JSON 문자열을 dict로 파싱한다.
        # 아주 드물게 LLM이 사용자 응답을 JSON으로 만들지 못하는 경우가 있는데,
        # 이 때 예외를 처리하기 위한 try catch이다.
        try:
            return json.loads(speak_dict_string)
        except json.JSONDecodeError as e:
            import logging
            logging.error(f"::Error Exception(JSONDecodeError):: 원문 Parsing 중 예외 발생!")
            logging.error(f"::에러 발생 원문:: {speak_dict_string}")
            logging.error(f"::예외 내용:: {e}")
            raise IncorrectAnswer("잘못된 형식의 응답입니다. 다시 한번 말해주세요.")  # 사용자가 다시 질문을 하도록 상위 함수로 예외 전달

    def split_invoke(self, message_history:str)->list:
        """
        전달 받은 문장들을 하나의 단일 의미나 사건으로 분리한다.
        :param message_history: 복합 의미를 가진 문장
        :return: 하나의 의미만을 가진 문장
        """
        import json
        # NOTE 1. 문장을 단일 의미로 분리한다.
        split_messages_string = self.__split_chain.invoke({"input": message_history, "date": datetime.date.today(), "example": self.__SPLIT_TEMPLATE_EXAMPLE}).content.strip()

        # NOTE 2. JSON 문자열을 list로 파싱한다.
        # 아주 드물게 LLM이 사용자 응답을 JSON으로 만들지 못하는 경우가 있는데,
        # 이 때 예외를 처리하기 위한 try catch이다.
        try:
            return json.loads(split_messages_string)["sentence"]
        except json.JSONDecodeError as e:
            import logging
            logging.error(f"::Error Exception(JSONDecodeError):: 원문 Parsing 중 예외 발생!")
            logging.error(f"::에러 발생 원문:: {split_messages_string}")
            logging.error(f"::예외 내용:: {e}")
            raise IncorrectAnswer("잘못된 형식의 응답입니다. 다시 저장해주세요.")

    __SPLIT_TEMPLATE = [
        ("system", "/json\n/no_think\n"),
        ("system", dedent("""
            당신의 임무는 human이 입력한 **복합 문장**을 의미 단위로 분리하여, 
            각각 **주어 + 술어(서술어) + 목적어/보어**가 포함된 **단일 사건 중심의 단문**으로 나누는 것이다.

            - 출력 형식 및 규칙:
            1. 출력은 반드시 아래 예시와 **동일한 JSON 구조**로 반환 (key는 `sentence`)
            2. 주석, 설명, 메타 텍스트는 **절대 포함 금지**
            3. `sentence` 키 아래에는 리스트(list) 형태로 문장들을 나열
            4. 각 문장은 [주어 + 서술어 + 목적어/보어] 형태의 **단일 사실**만 포함
            5. 주어나 목적어가 대명사(예: "그", "이것")일 경우, **문맥상 정확한 명사로 복원**
            6. 의미 없는 부사, 접속어(예: "그래서", "하지만")는 포함하지 않음
            7. 모든 문장 앞에 해당 사건에 맞는 날짜(예: "2025년 5월 6일")를 명시 (현재 날짜: {date})
            8. ai의 문장은 맥락을 이해하는데 **참고**하되, 문장 추출 절대 금지
            9. 아무런 정보가 없는 문장은 문장 추출에서 제외(예: 아 배고파, 뭐하지, 심심해 ...) 

            {example}
        """)),
        ("human", "{input}")
    ]

    __SPLIT_TEMPLATE_EXAMPLE = dedent("""
        - 예시:
        human: "세종대왕(1397–1450)은 조선의 제4대 왕으로서, 훈민정음을 창제하여 백성들이 쉽게 글을 익히도록 하였다. 세종대왕은 일을 하는 것을 좋아했는데, 이것 때문에 지병을 갖게 되었다."
        output:
        {
              "sentence": [
                "0000년 0월 0일, 세종대왕은 조선의 제4대 왕이다.",
                "0000년 0월 0일, 세종대왕은 훈민정음을 창제하였다.",
                "0000년 0월 0일, 세종대왕은 백성이 쉽게 글을 익히도록 하였다.",
                "0000년 0월 0일, 세종대왕은 일을 하는 것을 좋아했다.", 
                "0000년 0월 0일, 세종대왕은 일을 하는 것을 좋아해서 지병을 갖게 되었다."
              ]
        }
    """).strip()

    __SPEAK_TEMPLATE = [
        ("system", "/json\n/no_think\n"),
        ("system", dedent("""
                당신의 임무는 **입력 문장**에서 핵심 정보를 `ner` 형식
                `[주어, 목적어, 분리된 문장]` 로 **모두** 추출해 **유효한 JSON** 으로만 반환하는 것이다.

                - 규칙
                1. 출력은 반드시 아래 예시 구조와 **동일한 키·중괄호**를 사용.
                2. 주석, 설명, 불필요 텍스트 절대 포함 금지.
                3. 한 문장에 ner 이 여러 개면, 'triplets' 배열에 모두 나열.
                4. 주어 목적어가 없으면, 문장의 가장 중요한 단어로 대체. 
                5. 유효한 삼중항 혹은, JSON으로 반환하지 못하면, `반환 실패` 문구만 생성.

                - 예시
                {example}
        """).strip()),
        ("human", "{input}")
    ]

    SPEAK_TEMPLATE_EXAMPLE = dedent("""
            {
              "passage": "<전체 문장>",
              "triplets": [
                ["<주어>", "<목적어>", "<분리된 문장>"],
                ...
              ]
            }
            아래 예시를 참고해서 triplet을 작성하라.
            ---
            예시 1:
            {
              "passage": "세종대왕(1397–1450)은 조선의 제4대 왕으로서, 훈민정음을 창제하여 백성들이 쉽게 글을 익히도록 하였다. 그는 집현전을 설치하고 학자들을 등용하여 학문을 진흥시켰으며, 과학·농업·음악 등 다방면에서 국가의 발전을 이끌었다.",
              "triplets": [
                ["세종대왕", "조선의 제4대 왕", "세종대왕(1397-1450)은 조선의 제4대 왕이다."],
                ["세종대왕", "훈민정음", "세종대황은 훈민정음을 창제하였다."],
                ["세종대왕", "집현전", "세종대황은 집현전을 설치하였다."],
                ["세종대왕", "학자들", "세종대왕을 학자들을 등용하였다."],
                ["세종대왕", "학문", "세종대황은 학문을 진흥시켰다."],
                ["세종대왕", "국가의 발전", "세종대왕은 국가의 발전을 이끌었다."]
              ]
            }
            ---
            이제 다음 문장을 위 형식에 맞춰 작성해:
        """).strip()