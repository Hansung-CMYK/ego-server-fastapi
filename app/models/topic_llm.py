import json

from langchain_core.prompts import ChatPromptTemplate

from app.exception.exceptions import ControlledException, ErrorCode
from app.models.default_ollama_model import task_model


class TopicLlm:
    """
    Ollama를 통해 LLM 모델을 가져오는 클래스

    채팅내역을 바탕으로 일기를 생성하는 LLM 모델이다.
    """
    def __init__(self):
        prompt = ChatPromptTemplate.from_messages(self.__DIARY_TEMPLATE)
        self.__prompt = prompt | task_model

    def invoke(self, story: str) -> list[dict]:
        """
        일기를 생성하는 함수이다.
        """
        answer = self.__prompt.invoke({"input": story, "example": self.__EXAMPLE}).content
        try:
            diary = json.loads(answer)["topics"]
        except json.JSONDecodeError:
            raise ControlledException(ErrorCode.FAILURE_JSON_PARSING)
        except KeyError:
            raise ControlledException(ErrorCode.INVALID_DATA_TYPE)
        return diary

    __DIARY_TEMPLATE = [
        # 1) 메타 규칙
        ("system", "/no_think"),
        # 2) 작성 지침
        ("system", """
            <WRITING INSTRUCTIONS>
            1. **무조건** JSON **문자열**만 출력 (자연어 해설 금지)
            2 1인칭 일기체 사용(“나는 …했다.” “오늘은 …였다”)
            3. 최상위 키값은 'topics'만 존재.
                - 타입은 list로 저장. 
                - 타입 list 내부에는 주제가 dict 타입으로 저장 
            4. 키·값 모두 쌍따옴표("), 마지막 콤마 금지. 
            5. 주제(title) 제한
                - 각 title은 **1문장 이내** 핵심어
                - content : 3~10문장, 감각 묘사 ≥1문장
            6. 본문(content) 규칙
                - 가능하면 감정·환경 묘사(시각/청각/후각) 한 줄 삽입 
            7. 금지어: AI·챗봇·대화방·시스템·프롬프트 등 메타 표현
            8. 제공된 대화기록 외 새로운 사실은 채택하지 말 것.
            9. 작성된 시간을 고려해 다른 연관 정보도 함께 사용 가능 
            10. 하나의 주제도 도출하지 못했다면, empty list 반환
            11. 전체 구조는 SCHEMA EXAMPLE을 참고
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
    "topics": [
        {
            "title": "<주제 1>", 
            "content": "<본문 1>. <본문 2>..."
        },
        {
            "title": "<주제 2>",
            "content": "..."
        },
        ...
    ]
    """

topic_llm = TopicLlm()