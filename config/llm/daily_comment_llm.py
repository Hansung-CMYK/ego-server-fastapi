from textwrap import dedent

from app.internal.exception.error_code import ControlledException, ErrorCode
from app.internal.logger.logger import logger
from config.common.common_llm import CommonLLM


class DailyCommentLLM(CommonLLM):
    """
    요약:
        하루 한줄평을 생성하는 LLM

    설명:
        DiaryService에서 생성한 모든 정보를 기반으로 일기 한줄 요약을 생성한다.

    Attributes:
        __DAILY_COMMENT_TEMPLATE(tuple): 하루 한줄평을 생성하기 위한 시스템 프롬프트
        __RESULT_EXAMPLE(tuple): 하루 한줄평의 예시 프롬프트
    """

    __DAILY_COMMENT_TEMPLATE = ("system", dedent("""
        <PRIMARY_RULE>
        1. **Return valid JSON only** – no extra text before/after.  
        2. Do **NOT** output explanations, comments or system tags.  
        3. Keys & string values must use straight double-quotes (").
        </PRIMARY_RULE>

        <ROLE>
        • Summarise the given Korean `INPUT` sentences into **one single Korean sentence** (≤ 80 characters).  
        • This sentence becomes the user's “one-line diary”.
        </ROLE>

        <SUMMARY_REQUIREMENTS>
        • Base the summary on both chat logs & diary topics (“events”).  
        • Use **every** piece of information – nothing may be omitted.  
        • Merge the data smoothly so it feels natural in Korean.  
        • Include the following placeholders in the sentence (wrap each with single quotes `'`):  
            – `events`   (list of strings)  
            – `feelings` (list of strings)  
            – `keywords` (list of strings)  
        • If a placeholder list is empty, you may omit it and keep the sentence fluent.
        </SUMMARY_REQUIREMENTS>

        <OUTPUT_SCHEMA>
        ```json
        {{
          "result": "<summarised sentence>"
        }}
        ```
        </OUTPUT_SCHEMA>

        <EXAMPLE_OUTPUT>
        - {{"result": "~ 가 있었던 오늘, ~ 감정이 하루를 지배했고, ~ 가 곁을 맴돌았어요."}}
        - {{"result": "~ 일로 인해, ~ 한 하루였어요. ~ 가 기억에 남아요."}}
        - {{"result": "~ 과 함께, ~ 를 겪으며, ~ 을 느꼈어요}}
        </EXAMPLE_OUTPUT>

        <REFERENCE>
        {result_example}
        </REFERENCE>

        Q. <INPUT> events: {events}, feelings: {feelings}, keywords: {keywords} </INPUT>
        A.
        """))

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

    def __add_template(self) ->list[tuple]:
        return [self.__DAILY_COMMENT_TEMPLATE]

    def invoke(self, parameter:dict)->str:
        """
        요약:
            일기에서 생성한 값들을 조합해 하나의 문장을 만드는 함수

        Parameters:
            parameter(dict): parameter는 다음과 같은 key-value를 갖는다.
                - events(list[str]): 일기(있었던 일)
                - feelings(list[str]): 감정
                - keywords(list[str]): 키워드

        Raises:
            FAILURE_JSON_PARSING: JSON Decoding 실패 시, 빈 문자열 반환
                - {"result":str} 한줄 요약 실패 시, 형태로 반환이 실패하기 때문
        """
        parameter.update({
            "result_example":self.__RESULT_EXAMPLE
        })

        try:
            return super().invoke(parameter)
        except ControlledException(ErrorCode.FAILURE_JSON_PARSING):
            logger.exception(f"\n\nLLM이 일기 한줄 요약을 실패했습니다. 일기에 빈 문자열을 반환합니다.\n")
            return ""