import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


class DiaryLLMModel:
    def __init__(self, model_name="gemma3:4b"):
        model = ChatOllama(
            model=model_name,
            temperature=0.0,
            format="json"
        )

        prompt = ChatPromptTemplate.from_messages(self.__DIARY_TEMPLATE)
        self.__prompt = prompt | model

    def diary_invoke(self, story: str) -> list[dict]:
        answer = self.__prompt.invoke({"input": story, "example": self.__EXAMPLE}).content
        try:
            diary = json.loads(answer)
        except json.JSONDecodeError:
            raise Exception("-2: 일기 생성 중 JSON 변환이 실패되었습니다.")
        return diary

    __DIARY_TEMPLATE = [
        # 1) 메타 규칙
        ("system", "/no_think"),
        # 2) 작성 지침
        ("system", """
            <WRITING INSTRUCTIONS>
            1. **무조건** JSON **문자열**만 출력 (자연어 해설 금지)
            2 1인칭 일기체 사용(“나는 …했다.” “오늘은 …였다”)
            3. 키·값 모두 쌍따옴표("), 마지막 콤마 금지
            4. 주제(topic) 제한
                - 각 topic은 **1문장 이내** 핵심어
                - content : 3~10문장, 감각 묘사 ≥1문장
            5. 본문(content) 규칙
                - 가능하면 감정·환경 묘사(시각/청각/후각) 한 줄 삽입 
            6. 금지어: AI·챗봇·대화방·시스템·프롬프트 등 메타 표현
            7. 제공된 대화기록 외 새로운 사실은 채택하지 말 것.
            8. 작성된 시간을 고려해 다른 연관 정보도 함께 사용 가능 
            9. 하나의 주제도 도출하지 못했다면, empty dict 반환
            10. 전체 구조는 SCHEMA EXAMPLE을 참고

            </WRITING INSTRUCTIONS>

            <SCHEMA EXAMPLE>
            {example}
            </SCHEMA EXAMPLE>

            다음은 오늘의 대화 기록이다:
            """
         ),
        # 3) 사용자 입력
        ("human", "{input}"),
    ]

    __EXAMPLE = """
    [
        {
            topic: "<주제 1>", 
            content: "<본문 1>. <본문 2>..."
        },
        {
            topic: "<주제 2>",
            content: "..."
        },
        ...
    ]
    """


diary_llm = DiaryLLMModel()