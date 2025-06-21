from datetime import datetime
from textwrap import dedent

from app.internal.exception.error_code import ControlledException, ErrorCode
from config.common.common_llm import CommonLLM


class SplitLLM(CommonLLM):
    """
    요약:
        복합 문장을 단일 문장으로 분리하는 LLM

    설명:
        사용자 메세지에 담겨있는 의미들을 단일 문장으로 분리한다.

    Attributes:
        __SPLIT_TEMPLATE(tuple): 복합 문장을 분리하기 위한 시스템 프롬프트
        __RESULT_EXAMPLE(tuple): 복합 문장 분리의 예시 프롬프트
    """
    __SPLIT_TEMPLATE = ("system", dedent("""
        <PRIMARY_RULE>
        1. **Return valid JSON only** – nothing before/after the object.
        2. Every item in `"result"` **must** follow the pattern [Subject, Verb, Object/Complement].
        3. Extract & rewrite **HUMAN:** sentences _only_ – **never** include or split `AI:` lines.
        </PRIMARY_RULE>

        <ROLE>
        • Your task is to **split each complex Korean sentence** under `Q. HUMAN:`  
          into multiple single-fact sentences, each in the triple form [S, V, O/C].
        </ROLE>

        <SPLITTING_GUIDELINES>
        • Replace pronouns with their explicit noun referents.  
        • Drop filler adverbs / interjections / conjunctions (e.g. “와!”, “헐”, “그래서”, “하지만”).  
        • If a line has no informative content (e.g. “아 배고파”, “뭐하지”) → **skip** it.  
        • If the original sentence lacks an explicit first-person noun, use “나”.  
        • Append current timestamp `{datetime}` to the end of **every** output sentence  
          (format: `YYYY-MM-DDTHH:MM:SSSSS`).
        </SPLITTING_GUIDELINES>

        <OUTPUT_SCHEMA>
        ```json
        {{"result": [ "<sentence 1>", "<sentence 2>", ... ]}}
        ```
        </OUTPUT_SCHEMA>

        <EXAMPLE_RESULT>
        {result_example}
        </EXAMPLE_RESULT>

        Q. {input}
        A.
        """))

    __RESULT_EXAMPLE = dedent("""
    Q. AI: 그때 이야기했던 세종대왕에 대해 이야기 해줘.\nHUMAN: 그 사람은 조선의 제4대 왕으로서, 훈민정음을 창제하여 백성들이 쉽게 글을 익히도록 하였다. 특히 일을 하는 것을 좋아했는데, 이것 때문에 지병을 갖게 되었다.
    A. {"result": ["세종대왕은 조선의 제4대 왕이다. at 0000-00-00T00:00:000","세종대왕은 훈민정음을 창제하였다. at 0000-00-00T00:00:000","세종대왕은 백성이 쉽게 글을 익히도록 하였다. at 0000-00-00T00:00:000","세종대왕은 일을 하는 것을 좋아했다. at 0000-00-00T00:00:000", "세종대왕은 일을 하는 것을 좋아해서 지병을 갖게 되었다. at 0000-00-00T00:00:000"]}
    Q. AI: 어떻게 혜화에서 이렇게 빨리 여기까지 도착할 수 있었어?\nHUMAN: 교통 체증으로 인해 출발이 지연되어 우회했습니다. 거기는 보통 차가 자주 막히는 곳인데, 창신역으로 우회하면 막히지 않고 도착할 수 있습니다.
    A. {"result": ["나는 혜화에서 교통체증이 있었다. at 0000-00-00T00:00:000", "나는 교통체증으로 인해 출발이 지연되었다. at 0000-00-00T00:00:000","나는 교통체증을 해결하기 위해 우회했다. at 0000-00-00T00:00:000","혜화역은 차가 자주 막히는 곳이다. at 0000-00-00T00:00:000","차가 자주 막히는 혜화는 창신역으로 우회하면 막하지 않고 도착할 수 있다. at 0000-00-00T00:00:000"]}
    Q. AI: 좋은 영화 하나만 추천해 줘.\nHUMAN: 저는 어제 친구들과 ‘인터스텔라’를 봤습니다. 내용이 심오했고 음악이 감동적이었어요. 그래서 오늘도 그 사운드트랙을 계속 들었습니다.
    A. {"result": ["나는 어제 친구들과 ‘인터스텔라’를 보았다. at 0000-00-00T00:00:000","‘인터스텔라’의 내용은 심오하다. at 0000-00-00T00:00:000","‘인터스텔라’의 음악은 감동적이다. at 0000-00-00T00:00:000","나는 오늘 ‘인터스텔라’ 사운드트랙을 계속 들었다. at 0000-00-00T00:00:000"]}
    Q. AI: 저는 오늘 학교에서 양자역학 수업을 들었어요. 슈뢰딩거의 고양이에 대해서 배웠는데, 왜이리 어려운지 모르겠어요.\nHUMAN: 슈뢰딩거의 고양이 이론은 슈뢰딩거 상자를 열어 관측하기 전까지는 살아있는 고양이와 죽어있는 고양이가 상자 안에서 중첩된 상태로 공존한다는 이야기야. 이 실험은 원래 양자 역학의 불완전한 면을 비판하기 위해 만들어졌는데, 아이러니하게도 시간이 지나자 양자 역학을 묘사하는 가장 대표적인 사고 실험이 되어버렸어.
    A. {"result": ["슈뢰딩거의 고양이 이론은 상자를 열어 관측하기 전까지 살아 있는 고양이와 죽어 있는 고양이가 중첩 상태로 공존한다는 이론이다. at 0000-00-00T00:00:000","이 사고 실험은 양자 역학의 불완전함을 비판하기 위해 고안되었다. at 0000-00-00T00:00:000","슈뢰딩거의 고양이 사고 실험은 시간이 지나 대표적인 양자 역학 설명으로 자리 잡았다. at 0000-00-00T00:00:000"]}
    Q. AI: 점심 뭐 먹었어?\nHUMAN: 점심으로 김치찌개를 먹었고 너무 매워서 물을 많이 마셨어. 그런데 맛있어서 만족했어.
    A. {"result": ["나는 점심으로 김치찌개를 먹었다. at 0000-00-00T00:00:000","김치찌개가 매우 매웠다. at 0000-00-00T00:00:000","나는 매운 김치찌개 때문에 물을 많이 마셨다. at 0000-00-00T00:00:000","나는 매웠지만 김치찌개의 맛에 만족했다. at 0000-00-00T00:00:000"]}
    Q. AI: 오늘 날씨 어땠어?\nHUMAN: 아침에는 비가 왔지만 오후에는 해가 떠서 산책을 했어. 덕분에 기분이 좋아졌어.
    A. {"result": ["아침에는 비가 내렸다. at 0000-00-00T00:00:000","오후에는 해가 떴다. at 0000-00-00T00:00:000","나는 오후에 산책을 했다. at 0000-00-00T00:00:000","산책 덕분에 나는 기분이 좋아졌다. at 0000-00-00T00:00:000"]}
    """)

    def __add_template(self) ->list[tuple]:
        return [self.__SPLIT_TEMPLATE]

    def invoke(self, parameter:dict)->list[str]:
        """
        요약:
            전달 받은 문장들을 하나의 단일 의미나 사건으로 분리하는 함수

        Parameters:
            parameter(dict): parameter는 다음과 같은 key-value를 갖는다.
                - input(str): 분리할 복합 문장

        Raises:
            ControlledException(ErrorCode.FAILURE_SPLIT_MESSAGE): 분리할 문장 없음
        """
        parameter.update({
            "datetime": datetime.now().isoformat(timespec="milliseconds"),
            "result_example": self.__RESULT_EXAMPLE
        })

        split_messages = super().invoke(parameter)

        # 문장 분리 실패 시, 데이터는 저장하지 않는다.
        if len(split_messages) == 0:
            raise ControlledException(ErrorCode.FAILURE_SPLIT_MESSAGE)
        return split_messages