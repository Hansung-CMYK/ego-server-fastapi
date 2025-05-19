from langchain_core.prompts import ChatPromptTemplate

from app.models.default_model import task_model
import json
import logging

class DailyCommentLLM:
    def __init__(self):
        prompt = ChatPromptTemplate.from_messages(self.__DAILY_TEMPLATE)
        self.__daily_chain = prompt | task_model

    def invoke(self, diaries:list[dict] , feeling:list[str], keywords:list[str]):
        events = [diary["title"] for diary in diaries]
        diary_string = self.__daily_chain.invoke({"events": events, "feeling": feeling, "keywords":keywords, "example":self.__DAILY_TEMPLATE_EXAMPLE}).content.strip()

        try:
            return json.loads(diary_string)["comment"]
        except json.JSONDecodeError:
            logging.warning(f"LLM이 일기 한줄 요약을 실패했습니다.")
            logging.warning(f"원본 문장: {diary_string}")
            return ""

    __DAILY_TEMPLATE = [
        # 1) 체인-of-thought 차단
        ("system", "/no_think\n"),
        # 2) 역할 + 상세 규칙
        (
            "system",
            """
                <GOAL>
                대화·일기를 기반으로 ‘하루 한 줄 요약(comment)’을 작성한다.
                </GOAL>
                
                <OUTPUT RULES>
                1. **반드시** JSON **문자열** 하나만 출력하고, 최상위 키는 "comment" 고정
                2. 주석·설명 등 자연어 메타 텍스트 일절 금지
                3. 한글 **1문장만** 작성, **80자 이하** (마침표·느낌표 포함)
                4. 문장에 아래 요소를 **모두** 넣는다  
                   • events  ─> 제목들을 간단히 묶어 요약 (띄어쓰기 포함 15자 이내)  
                   • feeling ─> 리스트 내 단어를 전부 *'단일따옴표* 로 감싸서 사용  
                   • keywords─> 리스트 내 단어를 전부 *'단일따옴표* 로 감싸서 사용
                   • 만약 요소가 없다면, 해당 요소를 생략하고 자연스럽게 문장을 연결할 것 
                5. 사용 형식 예시  
                   {example}
                </OUTPUT RULES>
                
                <INPUT>
                    events   : {events}          # 리스트[str]
                    feeling  : {feeling}         # 리스트[str]
                    keywords : {keywords}        # 리스트[str]
                </INPUT>
            """
        ),
    ]

    __DAILY_TEMPLATE_EXAMPLE = """
    1. {"comment": "~ 가 있었던 오늘, ~ 감정이 하루를 지배했고, ~ 가 곁을 맴돌았어요."}
    2. {"comment": "~ 일로 인해, ~ 한 하루였어요. ~ 가 기억에 남아요."}
    3. {"comment": "~ 과 함께, ~ 를 겪으며, ~ 을 느꼈어요}
    """

daily_comment_llm = DailyCommentLLM()
