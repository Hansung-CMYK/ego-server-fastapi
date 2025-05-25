from json import JSONDecodeError
from textwrap import dedent
from langchain_core.prompts import ChatPromptTemplate
import json

from app.exception.exceptions import ControlledException, ErrorCode
from app.models.default_model import task_model, DEFAULT_TASK_LLM_TEMPLATE
from app.logger.logger import logger

class PersonaLlm:
    """
    요약:
        Persona 수정 사항을 생성하는 Ollama 클래스

    설명:
        대화 내역을 바탕으로 사용자의 페르소나를 재구성해주는 모델이다.

    Attributes:
        __chain: llm을 활용하기 위한 lang_chain
    """
    def __init__(self):
        # 랭체인 생성
        prompt = ChatPromptTemplate.from_messages(self.__PERSONA_TEMPLATE)
        self.__chain = prompt | task_model

    def persona_invoke(self, user_persona: dict, session_history: str) -> dict:
        """
        요약:
            사용자의 대화기록으로 페르소나를 수정하는 함수이다.

        Parameters:
            user_persona(dict): (변경될) 사용자 페르소나 dict 정보
            session_history(str): 사용자의 최근 대화 내역

        Raises:
            JSONDecodeError: JSON Decoding 실패 시, 빈 딕셔너리 반환
        """
        # 페르소나 변경사항
        answer:str = self.__chain.invoke({
            "session_history": session_history,
            "current_persona": user_persona,
            "return_form": self.__RETURN_FORM_EXAMPLE,
            "result_example": self.__RESULT_EXAMPLE,
            "default_task_llm_template": DEFAULT_TASK_LLM_TEMPLATE
        }).content
        clean_answer:str = self.__clean_json_string(text=answer) # 필요없는 문자열 제거

        # LOG. 시연용 로그
        logger.info(msg=f"\n\nPOST: api/v1/diary [페르소나 변경사항]\n{clean_answer}\n")

        # dict로 자료형 변경
        try:
            return json.loads(clean_answer)
        except JSONDecodeError:
            raise ControlledException(ErrorCode.FAILURE_JSON_PARSING)

    @staticmethod
    def __clean_json_string(text: str) -> str:
        """
        요약:
            LLM이 출력한 문자열에서 \```json 및 \``` 마커를 제거하고 공백을 정리한다.

        Parameters:
            text(str): 정제할 텍스트
        """
        text = text.strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text

    __PERSONA_TEMPLATE = [
        ("system", "/no_think {default_task_llm_template}"),
        ("system", dedent("""
        <PRIMARY_RULE>
        무조건 JSON 형식을 유지해야 합니다.
        JSON 외에 자연어 해설은 없습니다.
        시스템, 주석, 설명, 프롬프트 등 메타 표현은 절대 금지합니다.
        </PRIMARY_RULE>
        
        <ROLE>
        당신의 임무는 `Q.` 속 Human의 **Persona 변화**를 찾아내는 것 입니다.
        추가 및 삭제 되어야 할 값은 정해진 `key`로 반환합니다.
        </ROLE>
        
        <CURRENT_PERSONA>
        {current_persona}
        </CURRENT_PERSONA>
        
        <RULE>
        다음은 주어진 입력에 **필수적**으로 지켜야 할 반환 규칙입니다.
        - CURRENT_PERSONA는 변경이 될 JSON 정보입니다.
        - CURRENT_PERSONA를 참고해 각 값을 **삽입, 삭제** 정보를 추출합니다.
        </RULE>
        
        <RETURN_TYPE>
        - 출력은 반드시 아래 예시와 동일한 JSON 구조로 반환합니다.
        - 주어진 `key`를 제외하고 다른 키 값은 **무조건** 사용하면 안됩니다.
        - 주어진 `key`는 SAMPLE_JSON의 `key`만 사용할 수 있습니다.
        - 최상위 `key`는 `$set`과 `$unset`입니다.
        - 최상위 `key`인 `$set`과 `$unset`은 출력 값에 **무조건** 포함됩니다.
        - `$set`은 CURRENT_PERSONA에 **추가**될 정보입니다.
        - `$set` `key`의 `value`는 `dict`입니다.
        - `$set`은 CURRENT_PERSONA에 **삭제**될 정보입니다.
        - `$unset` `key`의 `value`는 `dict`입니다.
        - `dict`의 `key`는 무조건 'likes(좋아하는 것)', 'dislikes(싫어하는 것)', 'personality(성격)', personality(관심사), 'goal(목표)'만 가능합니다.
        - `dict`의 `key`는 **생성, 삭제 될 경우에만** 출력됩니다. 
        - 추가 및 삭제 되어야 할 값은 **정해진 `key` 값**을 유지해서 출력됩니다.
        </RETURN_TYPE>

        <RETURN_FORM>
        # 추가될 정보
        {return_form}
        </RETURN_FORM>
        
        <RESULT>
        {result_example}
        Q. <INPUT>{session_history}</INPUT>
        A. """))
    ]

    __RETURN_FORM_EXAMPLE = dedent("""
    "$set": {
        "likes": [<str, 좋아하는 것>, ...],
        "dislikes": [<str, 싫어하는 것>, ...]
        "personality": [<str, 성격>, ...],
        "goal": [str, 경제적 목표, ...]
    },
    # 삭제될 정보
    "$unset": {
        "likes": [<str, 좋아하는 것>, ...],
        "dislikes": [<str, 싫어하는 것>, ...]
        "personality": [<str, 성격>, ...],
        "goal": [str, 경제적 목표, ...]
    }
    """)

    __RESULT_EXAMPLE = dedent("""
    Q. <INPUT> U@Human: 요즘 암벽등반 시작했는데 손끝이 아플 만큼 재밌어! at 2025-05-21T19:12:14.002\nU@Human: 대신 커피는 입에 안 맞아서 끊었어. at 2025-05-21T19:12:45.110\nU@Human: 발표력 키워서 좀 더 외향적으로 변하고 싶다. at 2025-05-21T19:13:08.550 </INPUT>
    A. {"$set": {"likes": ["암벽등반"],"personality": ["외향적"],"goal": ["발표력 향상"]},"$unset": {"likes": ["커피"]}}
    Q. <INPUT> U@Human: 축구보다 요가가 몸에 더 잘 맞는 것 같아. at 2025-05-22T07:55:21.300\nU@Human: 더위도 너무 싫어서 여름엔 실내만 찾게 돼. at 2025-05-22T07:56:02.718\nU@Human: 취업 준비 본격적으로 시작해야지! at 2025-05-22T07:56:45.904 </INPUT>
    A. {"$set": {"likes": ["요가"],"dislikes": ["더위"],"goal": ["취업 준비"]},"$unset": {"likes": ["축구"]}}
    Q. <INPUT> U@Human: 이번 달엔 일찍 자고 새벽 러닝으로 상쾌하게 시작하려 해. at 2025-05-23T06:20:15.010\nU@Human: 단것은 줄이고 채소 위주 식단을 시도해 볼 거야! at 2025-05-23T06:21:02.447 </INPUT>
    A. {"$set": {"personality": ["규칙적인"],"goal": ["새벽 러닝 루틴"],"likes": ["채소"]},"$unset": {}}
    Q. <INPUT> U@Human: 솔직히 예전에는 공포영화를 무조건 챙겨 봤는데, 요즘은 잔인한 장면만 봐도 속이 불편해서 못 보겠어. at 2025-05-24T21:12:05.123\nU@Human: 대신 지난달부터 도자기 공방을 다니는데, 흙 만지다 보면 마음이 편안해져. at 2025-05-24T21:13:44.287\nU@Human: 내년엔 유럽 도자기 마을 투어를 가려고 적금도 들었지! at 2025-05-24T21:14:20.501\nU@Human: 사람들하고 자연스럽게 어울리는 편이 아니라서, 공방 친구들이랑 더 친해지고 싶어. at 2025-05-24T21:15:09.740 </INPUT>
    A. {"$set": {"likes": ["도자기 공예"],"dislikes": ["공포 영화"],"goal": ["유럽 도자기 마을 여행"],"personality": ["사교적"]},"$unset": {"likes": ["공포 영화"],"personality": ["내성적"]}}
    Q. <INPUT> U@Human: 매운 음식은 예전엔 위가 아파서 피했는데, 재작년부터 꾸준히 먹다 보니 이젠 청양고추도 씹어 먹을 정도로 좋아졌어. at 2025-05-25T19:02:11.555\nU@Human: 반면 집에서 게임만 하는 건 이젠 지루해서, 주말마다 보호소 봉사 나가고 있어. at 2025-05-25T19:03:26.909\nU@Human: 올해 목표는 영어 회화 실력을 확 끌어올려서 해외 봉사 프로그램에 지원하는 거야. at 2025-05-25T19:04:02.310\nU@Human: 그래서 게으름 피우는 습관은 꼭 버리고 싶어. at 2025-05-25T19:04:45.871 </INPUT>
    A. {"$set": {"likes": ["매운 음식", "봉사 활동"],"dislikes": ["게으름"],"goal": ["영어 회화 향상", "해외 봉사 참가"],"personality": ["규율적"]},"$unset": {"likes": ["게임"],"personality": ["게으른"]}}
    """)

# 싱글톤 생성
persona_llm = PersonaLlm()