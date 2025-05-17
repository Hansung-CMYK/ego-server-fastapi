import datetime
from textwrap import dedent

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

import json
import logging

class SplitLlm:
    """
    Ollama를 통해 LLM 모델을 가져오는 클래스
    """

    def __init__(self):
        model = ChatOllama(
            model="qwen3:8b",
            temperature=0.0,
            format="json"
        )

        # 문장 분리 프롬프트 적용 + 랭체인 생성
        split_prompt = ChatPromptTemplate.from_messages(self.__SPLIT_TEMPLATE)
        self.__split_chain = split_prompt | model

    def split_invoke(self, session_history:str)->list:
        """
        전달 받은 문장들을 하나의 단일 의미나 사건으로 분리한다.
        :param session_history: 복합 의미를 가진 문장
        :return: 하나의 의미만을 가진 문장
        """
        # NOTE 1. 문장을 단일 의미로 분리한다.
        split_messages_string = self.__split_chain.invoke({"input": session_history, "date": datetime.date.today(), "example": self.__SPLIT_TEMPLATE_EXAMPLE}).content.strip()

        # NOTE 2. JSON 문자열을 list로 파싱한다.
        # 아주 드물게 LLM이 사용자 응답을 JSON으로 만들지 못하는 경우가 있는데,
        # 이 때 예외를 처리하기 위한 try catch이다.
        try:
            return json.loads(split_messages_string)["sentence"]
        except json.JSONDecodeError:
            logging.warning(f"LLM이 대화내역을 단일문장으로 분리하지 못했습니다.")
            logging.warning(f"원본 문장: {split_messages_string}")
            return []

    __SPLIT_TEMPLATE = [
        ("system", "/json\n/no_think\n"),
        ("system", dedent("""
            당신의 임무는 human이 입력한 복합 문장을 의미 단위로 분리하여, 
            각각 주어 + 술어(서술어) + 목적어/보어가 포함된 단일 사건 중심의 단문으로 나누는 것이다.

            - 출력 형식 및 규칙:
            1. 출력은 반드시 아래 예시와 동일한 JSON 구조로 반환 (key는 `sentence`)
            2. 주석, 설명, 메타 텍스트는 절대 포함 금지
            3. `sentence` 키 아래에는 리스트(list) 형태로 문장들을 나열
            4. 각 문장은 [주어 + 서술어 + 목적어/보어] 형태의 단일 사실만 포함
            5. 문장은 육하원칙을 최대한 준수
            6. 모든 대명사는 원문 명사로 대체
            7. 의미 없는 부사, 접속어(예: "그래서", "하지만") 생략
            8. 모든 문장 앞에 해당 사건에 맞는 날짜(예: "0000년 0월 0일,")를 명시 (현재 날짜: {date})
            9. ai의 문장은 맥락을 이해하는데 참고하되, ai 답변은 절대 문장 추출 금지
            10. 아무런 정보가 없는 문장은 문장 추출에서 제외(예: 아 배고파, 뭐하지, 심심해 ...) 

            {example}
        """)),
        ("human", "{input}")
    ]

    __SPLIT_TEMPLATE_EXAMPLE = dedent("""
        - 예시:
        ai: 그때 이야기했던 세종대왕에 대해 이야기 해줘. 
        human: "그 사람은 조선의 제4대 왕으로서, 훈민정음을 창제하여 백성들이 쉽게 글을 익히도록 하였다. 특히 일을 하는 것을 좋아했는데, 이것 때문에 지병을 갖게 되었다."
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

split_llm = SplitLlm()