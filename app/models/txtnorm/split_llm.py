from textwrap import dedent

from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import json

from app.exception.exceptions import ControlledException, ErrorCode
from app.models.default_model import task_model, DEFAULT_TASK_LLM_TEMPLATE, clean_json_string
from app.logger.logger import logger

class SplitLlm:
    """
    요약:
        복합 문장을 단일 문장으로 분리하는 Ollama 클래스

    설명:
        사용자 메세지에 담겨있는 의미들을 단일 문장으로 분리하는 모델이다.

    Attributes:
        __chain: llm을 활용하기 위한 lang_chain
    """

    def __init__(self):
        # 문장 분리 프롬프트 적용 + 랭체인 생성
        split_prompt = ChatPromptTemplate.from_messages(self.__SPLIT_TEMPLATE)
        self.__chain = split_prompt | task_model

    def split_invoke(self, complex_sentence:str)->list:
        """
        요약:
            전달 받은 문장들을 하나의 단일 의미나 사건으로 분리하는 함수

        Parameters:
            complex_sentence(str): 복합 의미를 가진 문장

        Raises:
            JSONDecodeError: JSON Decoding 실패 시, 빈 리스트(`[]`) 반환
        """
        answer = self.__chain.invoke({
            "input": complex_sentence,
            "datetime": datetime.now().isoformat(),
            "result_example":self.__RESULT_EXAMPLE,
            "default_task_llm_template": DEFAULT_TASK_LLM_TEMPLATE
        }).content
        clean_answer: str = clean_json_string(text=answer)  # 필요없는 문자열 제거

        try:
            split_messages = json.loads(clean_answer)["result"]
        except json.JSONDecodeError:
            raise ControlledException(ErrorCode.FAILURE_SPLIT_MESSAGE)

        # 문장 분리 실패 시, 데이터는 저장하지 않는다.
        if len(split_messages) == 0:
            raise ControlledException(ErrorCode.FAILURE_SPLIT_MESSAGE)

        # LOG. 시연용 로그
        logger.info(msg=f"\n\nPOST: api/v1/chat [단일 문장 분리 성공]\n")

        return split_messages

    __SPLIT_TEMPLATE = [
        ("system", "/no_think {default_task_llm_template}"),
        ("system", dedent("""
        <PRIMARY_RULE>
        무조건 JSON 형식을 유지해야 합니다.
        무조건 `A.`는 [주어, 술어, 목적어/보어]를 가져야 합니다.
        응답은 `HUMAN:`의 문장만 반환합니다.
        </PRIMARY_RULE>
        
        <ROLE>
        당신의 임무는 `Q. HUMAN: ` 복합 문장을 **여러 문장으로 분리**하는 것 입니다.
        문장을 분리할 땐, 단일 사건이나 단일 의미를 [주어, 술어, 목적어/보어] 형태로 만들어야 합니다. 
        </ROLE>

        <RULE>
        다음은 주어진 입력에 **필수적**으로 지켜야 할 반환 규칙입니다.
        - 원본 문장에 있는 대명사는 모두 동일한 의미를 가진 명사로 대체합니다.
        - 원본 문장에 있는 의미없는 부사, 감탄사(예: '와!', '헐'), 접속어(예: '그래서', '하지만')은 제거합니다.
        - 분리된 각 문장은 [주어, 술어, 목적어/보어] 형태의 단일 사건이나 의미입니다.
        - 분리된 모든 문장 뒤에는 현재 시간({datetime})을 명시합니다. (예:'0000-00-00T00:00:000')
        - **`HUMAN:`의 문장만** 반환해야 합니다.
        - `AI:` 문장은 대화 맥락을 이해하는데 참고합니다.  
        - `AI:` 문장은 분리를 **절대** 금지합니다.
        - `AI:` 문장은 반환을 **절대** 금지합니다.
        - 아무런 정보가 없는 문장은 문장 분리를 제외합니다. (예: 아 배고파, 뭐하지, 심심해 ...)
        - 아무런 정보가 없는 문장은 문장 반환을 제외합니다. (예: 아 배고파, 뭐하지, 심심해 ...)
        - 본인을 지칭하는 명사가 문장에 없다면, `나`를 사용합니다.
        </RULE>
        
        <RETURN_TYPE>
        - 출력은 반드시 아래 예시와 동일한 JSON 구조로 반환합니다.
        - 최상위 key는 `result`입니다.
        - `result` `key`의 `value`는 `list[str]` `type`입니다. 
        - `result` `key`의 `value`는 분리된 여러 문장들이 저장됩니다.
        </RETURN_TYPE>

        <RESULT>
        {result_example}
        Q. {input}
        A. """)),
    ]

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

split_llm = SplitLlm()