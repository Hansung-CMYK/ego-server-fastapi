from textwrap import dedent

from app.internal.exception.error_code import ControlledException, ErrorCode
from app.internal.logger.logger import logger
from config.common.common_llm import CommonLLM


class PersonaLLM(CommonLLM):
    """
    요약:
        Persona 수정 사항을 생성하는 LLM

    설명:
        대화 내역을 바탕으로 사용자의 페르소나를 재구성한다.

    Attributes:
        _PERSONA_TEMPLATE(tuple): 사용자의 페르소나를 갱신하기 위한 시스템 프롬프트
        _RESULT_EXAMPLE(tuple): 페르소나 갱신 예시 프롬프트
    """

    _PERSONA_TEMPLATE = ("system", dedent("""
        <PRIMARY_RULE>
        1. **Return strictly valid JSON only.**
        2. Do **NOT** output any natural-language commentary, markdown, system tags, or explanations.
        </PRIMARY_RULE>

        <ROLE>
        • Your job is to detect **persona changes** for the Human speaker inside <INPUT>  
          and express them as JSON patches to the existing persona.
        • All Human utterances are in **Korean**, but you must still follow these English instructions exactly.
        </ROLE>

        <CURRENT_PERSONA>
        {current_persona}
        </CURRENT_PERSONA>

        <GUIDELINES>
        • Treat CURRENT_PERSONA as the baseline document to be updated.  
        • Compare the new chat log and list the attributes that should be **added to** or **removed from** the persona.
        </GUIDELINES>

        <OUTPUT_SCHEMA>
        • Use only the keys that appear in the SAMPLE section below.  
        • Top-level keys **must** be "$set" and "$unset" (both present, even if empty).  
        • Allowed attribute keys inside those objects:  
            - "likes"          (things the user likes)  
            - "dislikes"       (things the user dislikes)  
            - "personality"    (personality traits)  
            - "goal"           (goals)  
        • If there is nothing to add/remove for a given attribute, omit that attribute key.
        </OUTPUT_SCHEMA>

        <SAMPLE_JSON>
        {{ 
            "result": {{
                "$set": {{
                    "likes": [<str, 좋아하는 것>, ...],
                    "dislikes": [<str, 싫어하는 것>, ...]
                    "personality": [<str, 성격>, ...],
                    "goal": [str, 경제적 목표, ...]
                }},
                "$unset": {{
                    "likes": [<str, 좋아하는 것>, ...],
                    "dislikes": [<str, 싫어하는 것>, ...]
                    "personality": [<str, 성격>, ...],
                    "goal": [str, 경제적 목표, ...]
                }} 
            }} 
        }}
        </SAMPLE_JSON>

        <EXAMPLE_RESULT>
        {result_example}
        </EXAMPLE_RESULT>

        Q. <INPUT>{session_history}</INPUT>
        A.
        """))

    _RESULT_EXAMPLE = dedent("""
        Q. <INPUT> U@Human: 요즘 암벽등반 시작했는데 손끝이 아플 만큼 재밌어! at 2025-05-21T19:12:14.002\nU@Human: 대신 커피는 입에 안 맞아서 끊었어. at 2025-05-21T19:12:45.110\nU@Human: 발표력 키워서 좀 더 외향적으로 변하고 싶다. at 2025-05-21T19:13:08.550 </INPUT>
        A. {"result":{"$set": {"likes": ["암벽등반"],"personality": ["외향적"],"goal": ["발표력 향상"]},"$unset": {"likes": ["커피"]}}}}
        Q. <INPUT> U@Human: 축구보다 요가가 몸에 더 잘 맞는 것 같아. at 2025-05-22T07:55:21.300\nU@Human: 더위도 너무 싫어서 여름엔 실내만 찾게 돼. at 2025-05-22T07:56:02.718\nU@Human: 취업 준비 본격적으로 시작해야지! at 2025-05-22T07:56:45.904 </INPUT>
        A. {"result":{"$set": {"likes": ["요가"],"dislikes": ["더위"],"goal": ["취업 준비"]},"$unset": {"likes": ["축구"]}}}
        Q. <INPUT> U@Human: 이번 달엔 일찍 자고 새벽 러닝으로 상쾌하게 시작하려 해. at 2025-05-23T06:20:15.010\nU@Human: 단것은 줄이고 채소 위주 식단을 시도해 볼 거야! at 2025-05-23T06:21:02.447 </INPUT>
        A. {"result":{"$set": {"personality": ["규칙적인"],"goal": ["새벽 러닝 루틴"],"likes": ["채소"]},"$unset": {}}}
        Q. <INPUT> U@Human: 솔직히 예전에는 공포영화를 무조건 챙겨 봤는데, 요즘은 잔인한 장면만 봐도 속이 불편해서 못 보겠어. at 2025-05-24T21:12:05.123\nU@Human: 대신 지난달부터 도자기 공방을 다니는데, 흙 만지다 보면 마음이 편안해져. at 2025-05-24T21:13:44.287\nU@Human: 내년엔 유럽 도자기 마을 투어를 가려고 적금도 들었지! at 2025-05-24T21:14:20.501\nU@Human: 사람들하고 자연스럽게 어울리는 편이 아니라서, 공방 친구들이랑 더 친해지고 싶어. at 2025-05-24T21:15:09.740 </INPUT>
        A. {"result":{"$set": {"likes": ["도자기 공예"],"dislikes": ["공포 영화"],"goal": ["유럽 도자기 마을 여행"],"personality": ["사교적"]},"$unset": {"likes": ["공포 영화"],"personality": ["내성적"]}}}
        Q. <INPUT> U@Human: 매운 음식은 예전엔 위가 아파서 피했는데, 재작년부터 꾸준히 먹다 보니 이젠 청양고추도 씹어 먹을 정도로 좋아졌어. at 2025-05-25T19:02:11.555\nU@Human: 반면 집에서 게임만 하는 건 이젠 지루해서, 주말마다 보호소 봉사 나가고 있어. at 2025-05-25T19:03:26.909\nU@Human: 올해 목표는 영어 회화 실력을 확 끌어올려서 해외 봉사 프로그램에 지원하는 거야. at 2025-05-25T19:04:02.310\nU@Human: 그래서 게으름 피우는 습관은 꼭 버리고 싶어. at 2025-05-25T19:04:45.871 </INPUT>
        A. {"result":{"$set": {"likes": ["매운 음식", "봉사 활동"],"dislikes": ["게으름"],"goal": ["영어 회화 향상", "해외 봉사 참가"],"personality": ["규율적"]},"$unset": {"likes": ["게임"],"personality": ["게으른"]}}}
        """)

    def _add_template(self)->list[tuple]:
        return [self._PERSONA_TEMPLATE]

    def invoke(self, parameter:dict)->dict:
        """
        요약:
            사용자의 대화기록으로 페르소나를 수정하는 함수

        Parameters:
            parameter(dict): parameter는 다음과 같은 key-value를 갖는다.
                - session_history(list[str]): FastAPI가 실행된 후, 채팅방의 전체 대화 내역
                - current_persona(dict): 대화 내역 본인의 에고 페르소나

        Raises:
            JSONDecodeError: JSON Decoding 실패 시, 빈 딕셔너리 반환
        """
        parameter.update({
            "result_example": self._RESULT_EXAMPLE
        })
        try:
            return super().invoke(parameter)
        except ControlledException:
            logger.exception(f"\n\nPersona 변화 감지를 실패했습니다. 딕셔너리를 반환합니다.\n")
            return {}
