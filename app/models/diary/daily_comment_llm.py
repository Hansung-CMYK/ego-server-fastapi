from textwrap import dedent

from langchain_core.prompts import ChatPromptTemplate

from app.models.default_model import task_model
import json
from app.logger.logger import logger

class DailyCommentLLM:
    """
    요약:
        하루 한줄평을 생성하는 Ollama 클래스

    설명:
        일기 내용을 바탕으로 오늘 하루 한줄평을 생성하는 모델이다.

    Attributes:
        __chain: llm을 활용하기 위한 lang_chain
    """
    def __init__(self):
        # 랭체인 생성
        prompt = ChatPromptTemplate.from_messages(self.__DAILY_TEMPLATE)
        self.__chain = prompt | task_model

    def daily_comment_invoke(self, diaries:list[dict], feeling:list[str], keywords:list[str])->str:
        """
        요약:
            일기에서 생성한 값들을 조합해 하나의 문장을 만드는 함수

        Parameters:
            diaries(list[dict]): 각 주제 별로 분리된 일기 내용
            feeling(list[str]): 일기로 도출된 감정
            keywords(list[str]): 대화에서 많이 사용한 키워드

        Raises:
            JSONDecodeError: JSON Decoding 실패 시, 빈 문자열 반환
        """
        events = [diary["title"] for diary in diaries] # 일기에서 제목(주제)를 정제한다.

        diary_string = self.__chain.invoke({ # 한줄평 추출
            "events": events, "feelings": feeling, "keywords":keywords,
            "return_form_example":self.__RETURN_FORM_EXAMPLE,
            "result_example":self.__RESULT_EXAMPLE
        }).content.strip()

        # LOG. 시연용 로그
        logger.info(msg=f"\n\nPOST: api/v1/diary [한줄 요약 문장 생성]\n{keywords}\n")

        # 반환된 문자열 dict로 변환
        try:
            return json.loads(diary_string)["result"]
        except json.JSONDecodeError:
            logger.exception(f"\n\nLLM이 일기 한줄 요약을 실패했습니다.\n")
            return ""

    __DAILY_TEMPLATE = [
        ("system", dedent("""/no_think
        
        You have access to functions. If you decide to invoke any of the function(s),
        you MUST put it in the format of
        {"name": function name, "parameters": dictionary of argument name and its value}

        You SHOULD NOT include any other text in the response if you call a function
        """)),
        ("system", dedent("""
        <PRIMARY_RULE>
        무조건 JSON 형식을 유지해야 합니다.
        JSON 외에 자연어 해설은 없습니다.
        AI, 챗봇, 대화방, 시스템, 주석, 설명, 프롬프트 등 메타 표현은 절대 금지합니다.
        </PRIMARY_RULE>
        
        <ROLE>
        당신의 임무는 INPUT에 있는 문장들을 한 문장으로 요약하는 것입니다.
        반환되는 문장을 80자 이하입니다.
        </ROLE>
        
        <RULE>
        다음은 주어진 입력에 **필수적**으로 지켜야 할 반환 규칙입니다.
        - 대화와 일기를 기반으로 ‘하루 한 줄 요약’을 작성한다.
        - INPUT 정보를 한 문장으로 요약합니다.
        - INPUT 정보를 자연스럽게 연결해야 합니다.
        - INPUT 정보는 빠짐없이 **전부** 이용해야 합니다.
        - 출력은 한글로 문장으로 작성합니다. 문장은 80자 이하입니다.
        </RULE>
        
        <RETURN_TYPE>
        - 출력은 반드시 아래 예시와 동일한 JSON 구조로 반환합니다.
        - 최상위 `key`는 `result`입니다.
        - `result` `key`의 `value`는 `str` `type`입니다.
        </RETURN_TYPE>
        
        <WRITING_INSTRUCTIONS>
        다음은 문장을 요약할 때, 지켜야 할 요약문 작성 규칙입니다.
        - 요약된 문장에는 **무조건** 매개변수인 `events`, `feelings`, `keywords` 들을 **모두** 추가해야 합니다.
        - 각 매개변수들의 의미는 다음과 같습니다:
            * events(list[str]): 오늘 일기에 작성된 사건입니다.
            * feelings(list[str]): 오늘 작성자가 느꼈던 감정입니다. feelings의 값들은 단일따옴표(`'`) 로 감싸서 사용해야 합니다.
            * keywords(list[str]): 오늘 자주 거론된 단어입니다. keywords의 값들은 단일따옴표(`'`) 로 감싸서 사용해야 합니다.
        </WRITING_INSTRUCTIONS>
        
        <RETURN_FORM>
        다음은 출력 정보 예시입니다. 형식에 유사한 답변을 반환합니다.
        {return_form_example}
        - 만약 매개변수가 부족하여 문장을 완성할 수 없다면, 해당 요소를 생략하고 자연스럽게 문장을 연결합니다.
        </RETURN_FORM>
        
        <RESULT>
        {result_example}
        Q. <INPUT> events: {events}, feelings: {feelings}, keywords: {keywords} </INPUT>
        A. """))
    ]

    __RETURN_FORM_EXAMPLE = dedent("""
    - {"result": "~ 가 있었던 오늘, ~ 감정이 하루를 지배했고, ~ 가 곁을 맴돌았어요."}
    - {"result": "~ 일로 인해, ~ 한 하루였어요. ~ 가 기억에 남아요."}
    - {"result": "~ 과 함께, ~ 를 겪으며, ~ 을 느꼈어요}
    """)

    __RESULT_EXAMPLE = dedent("""
    Q. <INPUT> events: ['군대 동기와의 만남', '수업 지각'], feelings: ['기쁨', '화남'], keywords: ['군대', '약속 시간', '달리기'] </INPUT>
    A. {"result": "'군대 동기와의 만남'과 '수업 지각'이 있었던 오늘, '기쁨'과 '화남' 속에서, '군대', '약속 시간', '달리기'가 기억에 남아요!"}
    Q. <INPUT> events: ['체육시간 족구공으로 축구하기'], feelings: ['슬픈'], keywords: ['체육시간', '족구공'] </INPUT>
    A. {"result": "오늘은 '슬픈' 하루였어요. '체육시간'에 '족구공'으로 '체육시간 족구공으로 축구하기'가 있었기 때문이에요!"}
    Q. <INPUT> events: ['원두 시향', '헬스장 러닝'], feelings: ['뿌듯함', '상쾌함'], keywords: ['커피', '러닝머신', '땀'] </INPUT>
    A. {"result": "'원두 시향'과 '헬스장 러닝'으로 채운 오늘, '뿌듯함'과 '상쾌함'이 번지고 '커피', '러닝머신', '땀'이 잔향처럼 남았어요."}
    Q. <INPUT> events: ['부산 불꽃쇼'], feelings: ['설렘', '피곤'], keywords: ['KTX'] </INPUT>
    A. {"result": "'부산 불꽃쇼' 덕분에 '설렘'과 '피곤'이 뒤섞였고, 길 위에선 'KTX'가 하루를 잇는 실이었어요."}
    Q. <INPUT> events: ['팀장 피드백', 'KPI 수정', '야근'], feelings: [], keywords: ['PPT', '커피'] </INPUT>
    A. {"result": "'팀장 피드백', 'KPI 수정', '야근'이 이어진 밤, 'PPT'와 '커피'만이 고단함을 붙잡아 줬어요."}
    Q. <INPUT> events: ['아침 무지개'], feelings: ['잔잔함'], keywords: [] </INPUT>
    A. {"result": "'아침 무지개'를 만난 덕분에 하루 종일 '잔잔함'이 마음을 물들였어요."}
    Q. <INPUT> events: ['친구 생일파티', '깜짝 케이크', '늦은 귀가'], feelings: ['기쁨'], keywords: ['초', '노래'] </INPUT>
    A. {"result": "'친구 생일파티', '깜짝 케이크', '늦은 귀가'까지, '기쁨'이 번지고 '초', '노래'가 귓가에 남았어요."}
    Q. <INPUT> events: ['폭우 퇴근길', '버스 지연'], feelings: [], keywords: ['우산', '물웅덩이'] </INPUT>
    A. {"result": "'폭우 퇴근길'과 '버스 지연' 속에서도 '우산', '물웅덩이'가 익숙한 배경음이었어요."}
    """)

daily_comment_llm = DailyCommentLLM()
