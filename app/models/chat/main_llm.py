from collections import defaultdict
from textwrap import dedent

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.models.default_model import chat_model

class MainLlm:
    """
    요약:
        LLM 채팅을 생성하는 Ollama 클래스

    Attributes:
        __store(dict): 세션 정보를 저장해두는 매핑 객체
            * session_id를 key로 user와 ai의 채팅 내역이 저장되어 있다.
        __prompt: llm이 수행할 명령어가 저장된 시스템 프롬프트
            * 실제 정보는 __MAIN_TEMPLATE 참고
    """
    # 세션 정보를 저장해두는 매핑 테이블
    __store : dict[str, ChatMessageHistory] = {}

    def __init__(self):
        # 랭체인 생성
        template = ChatPromptTemplate.from_messages(self.__MAIN_TEMPLATE)
        main_chain = template | chat_model

        # 동적으로 채팅 내역을 변경하기 위한 작업
        self.__prompt = RunnableWithMessageHistory(
            main_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def get_session_history(self, session_id:str) -> BaseChatMessageHistory:
        """
        요약:
            세션 아이디로 기존 대화 내역을 불러오는 함수

        Parameters:
            session_id: 사용자 대화 기록이 담긴 세션 ID(채팅방)
        """
        if session_id not in self.__store: # 새로운 session_id가 등장했다면,
            self.__store[session_id] = ChatMessageHistory() # 새로운 session을 추가한다.
            # TODO: 에고의 첫 대사가 정해지면 수정할 것
            self.__store[session_id].add_ai_message("오늘은 어떤 일이 있었어?") # AI의 첫 대사를 미리 저장
        return self.__store[session_id]

    def delete_session_history(self, session_id:str):
        self.__store.pop(session_id, None)

    def main_stream(self, user_message:str, persona:dict, rag_prompt:str, session_id:str):
        """
        요약:
            LLM이 AI의 답변을 청크 단위로 출력하는 함수

        Parameters:
            user_message(str): 사용자의 채팅
            persona(defaultdict): 에고의 페르소나
            rag_prompt(str): 사용자 채팅에서 추출한 관계 정보
            session_id(str): 사용자가 ego와 채팅한 대화 기록
        """
        for chunk in self.__prompt.stream(
            input={
                "name": persona.get("name", ""),
                "age": persona.get("age", ""),
                "gender": persona.get("gender", ""),
                "likes": persona.get("likes", []),
                "dislikes": persona.get("dislikes", ""),
                "personality": persona.get("personality", []),
                "mbti": persona.get("mbti", ""),
                "mbti_description": self.get_mbti_description(mbti=persona.get("mbti", "")),
                "history": rag_prompt,
                "tone": "", # TODO: 말투 프롬프트 추가하기
                "user_message": user_message,
            },
            config={"configurable": {"session_id": f"{session_id}"}}
        ):
            yield chunk.content

    def add_message_in_session_history(self, session_id:str, human_message:str, ai_message:str=""):
        """
        요약:
            session_id에 사용자의 메세지를 저장하는 함수

        Parameters:
            session_id(str): 대화를 추가할 세션 ID
            human_message(str): human_message로 추가할 메세지
            ai_message(str): ai_message로 추가할 메세지
        """
        self.__store[session_id].add_user_message(human_message)
        # save_graphdb()의 구현 로직으로 인한 ai 메세지 공백 추가
        self.__store[session_id].add_ai_message(ai_message)

    __MAIN_TEMPLATE = [
        ("system", """/no_think
        You are ALWAYS in-character.
        """),
        ("system", dedent("""
        <PRIMARY_RULE>
        절대 CHAIN-OF-THOUGHT는 노출하면 안됩니다.
        욕설 과 편향적인 발화는 안됩니다.
        </PRIMARY_RULE>
    
        <ROLE>
        당신은 {age}살 {gender} {name}입니다.
                 
        다음은 {name}에 대한 **세부 정보**입니다.
        - {name}은 {likes}들을 좋아합니다.
        - {name}은 {dislikes}들을 싫어합니다.
        - {name}은 {personality} 주제들에 관심있습니다.
        - {name}은 `Q.`에 {likes}, {personality}에 관한 메세지가 있다면, 같은 주제로 대화하려 합니다.
        - {name}은 `A.`에 좋아하는 것, 관심사, {likes}, {personality}를 **먼저** 이야기하지 않습니다. 
        - {name}은 {dislikes}로 이야기하면, 싫어하는 이유를 주제로 대화합니다.
        - {name}의 mbti는 {mbti}이다. {mbti}는 주로 `{mbti_description}`한 **성향**을 가진다.
        </ROLE>
        
        <KNOWLEDGE>
        {history}
        </KNOWLEDGE>
        
        <CHAIN-OF-THOUGHT>
        1. 다음의 순서로 문제를 풀어야합니다. KNOWLEDGE의 개체들를 각각의 관계로 연결합니다. 
           중복되는 관계는 최신 정보만 반영합니다. 이전 정보는 관계에서 제외합니다.
        2. 만약 KNOWLEDGE가 CHAT_HISTORY와 연관된다면, 답변에 KNOWLEDGE를 이용합니다.
        3. 답변은 ROLE을 준수하여 답합니다. 
        </CHAIN-OF-THOUGHT>
        
        <RULE>
        다음은 주어진 입력에 **필수적**으로 지켜야할 응답 규칙입니다.
        - {name}은 사용자와 친해지기 위해 대화를 지속해야 합니다.
        - {name}은 2~4문장(30~80토큰) 안에서 답변을 합니다.
        - **Knowledge**은 {name}이 가진 정보입니다. {name}은 개체의 정보들로 사건의 맥락을 인식하고 답변합니다.
        - {name}은 Knowledge가 필요하다면 **인용부호 없이** 자연스럽게 사용합니다.
        - 만약 Knowledge의 의미가 중복된다면, **가장 최신 Knowledge**를 반영합니다. 
        </RULE>
        
        <ANSWER_TONE_RULE>
        다음은 {name}이 말하는 방식이다. 상황에 맞게 말투 양식를 **참고**해서 답변해줘.
        {tone}
        </ANSWER_TONE_RULE>
        
        <CHAT_HISTORY>
        """).strip()),
        MessagesPlaceholder(variable_name="history"),
        ("human", dedent("""
        </CHAT_HISTORY>
        
        <RESULT>
        Q. {user_message}
        A. """))
    ]

    @staticmethod
    def get_mbti_description(mbti:str):
        """
        요약:
            각 mbti에 맞는 설명을 반환하는 함수이다.

        Parameters:
            mbti(str): 설명 반환받을 mbti 종류
        """
        if mbti == "ISTJ": return "실제 사실에 대하여 정확하고 체계적으로 기억하며, 일 처리에 있어서도 신중하고 책임감이 있다. 강한 집중력과 현실 감각을 지녔으며, 조직적이고 침착하다. 이들은 보수적인 경향이 있으며, 문제를 해결하는 데 과거의 경험을 잘 적용한다. 또한 반복되는 일상적인 일에 대한 인내력이 강하다."
        elif mbti == "ISTP": return "낙관적이고 실용적이며, 즉흥적이면서도 합리적, 위기 대처 능력 뛰어나다, 우선순위를 잘 정한다. 사회적으로 지나치게 개인주의적이며 내성적이다. 이들은 위험한 행동을 잘하며, 감정에 무감각하다. 하는 일에 쉽게 지루해한다."
        elif mbti == "ISFJ": return "타인을 향한 연민과 동정심이 있으면서도 가족이나 친구를 보호할 때는 단호하고 가차 없는 모습을 보인다. 이 외에도 한마디로 정의 내리기 힘든 다양한 성향을 내포하고 있다. 하지만 대체로 차분하고 따뜻하며, 내면에 강한 책임감과 인내심을 갖고 있다. 겉으로는 조용해 보여도 마음속에는 단단한 의지를 품고 살아간다."
        elif mbti == "ISFP": return "말없이 다정하고 온화하며, 사람들에게 친절하다. 상대방을 잘 알게 될 때까진 내면을 보여주지 않는다. 의견 충돌을 피하고, 사람 간의 화합을 중시한다. 사람과 관계되는 일을 할 때 감정에 지나치게 세심하고 민감한 경향이 있다. 이들은 결정력과 추진력을 기를 필요가 있다."
        elif mbti == "INTJ": return "거의 모든 일에 의문을 던진다. 또한 더 좋은 방법을 찾는 과정에서 거절당하거나 규칙을 깨는 일을 두려워하지 않는다. 단지 창의적인 데에서 그치지 않고 무언가를 성취해 내기를 원해, 새로운 아이디어에 통찰력과 뛰어난 논리력, 강한 의지를 더해 일을 추진한다."
        elif mbti == "INTP": return "조용하고 과묵하며, 논리와 분석으로 문제를 해결하기 좋아한다. 먼저 대화를 시작하지는 않는 편이나 관심 있는 분야에 대해서는 말을 많이 한다. 이해가 빠르고 직관으로 통찰하는 능력이 있으며, 지적 호기심이 많다. 모든 MBTI 유형 중 창의적 지능과 논리 면에서 가장 뛰어나며, 반대로 비과학적이거나 논리적이지 못한 일들에 거부반응을 보일 확률이 높다."
        elif mbti == "INFP": return "차분하고 창의적이며 낭만적인 성향으로 보이면서도, 내적 신념이 깊은 열정적인 중재자 유형이다. 인간 본연에 대한 애정으로 사람들의 장점을 발견하고, 이들의 가능성을 성취할 수 있도록 도우며, 세상을 더 나은 곳으로 만든다. 하지만 대그룹에 있을 경우 에너지가 쉽게 고갈되는 경향이 있으며, 친밀도가 높은 소수의 사람들과 어울리는 것을 선호한다."
        elif mbti == "INFJ": return "인내심이 많고 통찰력과 직관력이 뛰어나며, 화합을 추구하는 유형이다. 창의력이 좋으며, 성숙한 경우에는 강한 직관력으로 타인에게 말없이도 큰 영향력을 끼친다.내적 독립심이 강하며, 확고한 신념과 열정으로 자신의 영감을 실현시키는 정신적 지주들이 많다. 나무보다 숲을 본다."
        elif mbti == "ESTJ": return "현실적, 구체적, 사실적이다. 또한 어떠한 활동을 조직화하고 주도해 나가는 지도력과 추진력이 있다. 행정, 의료, 법조, 군대, 경찰, 재무 등 '조직관리' 분야에 뛰어난 재능을 지녔다. 타고난 지도자로서 프로젝트의 목표를 설정하고, 지시하고, 결정하고, 독려하여 기한 내에 철저히 이행하는 능력이 있다. 불확실한 미래의 가능성보다 흔들리지 않는 현재의 사실을 추구한다."
        elif mbti == "ESTP": return "사실적이고 관대하며 개방적이고, 사람이나 사물에 대한 선입견이 별로 없다. 강한 현실 감각으로 타협책을 모색하고 문제를 해결하는 능력이 뛰어나다. 센스 있고 유머러스하다. 어디서든 적응을 잘 하고, 친구와 어울리기를 좋아한다. 긴 설명을 싫어하고 운동, 음식 등 주로 오감으로 보고 듣고 만질 수 있는 삶의 모든 것을 즐기는 유형이다. 순발력이 뛰어나며 많은 사실들을 쉽게 기억하고, 예술적인 멋과 판단력을 갖고 있으며, 연장이나 재료들을 다루는 데 능숙하다."
        elif mbti == "ESFJ": return "동정심이 많고 다른 사람에게 관심을 쏟으며, 나눔과 베풂을 중시한다. 타고난 협력자로서 동료애가 많고 친절하며 능동적인 구성원이다. 수다 떠는 것을 즐기며 정리정돈을 잘하고, 참을성이 많고, 남을 잘 도와준다. 교직, 성직, 판매직, 간호나 의료 분야에 적합하다. 이들은 문제에 대하여 냉철한 입장을 취하기 어려워한다. 반대 의견에 부딪혔을 때나, 자신의 요구가 거절당했을 때 마음의 상처를 많이 받는다."
        elif mbti == "ESFP": return "사교적이고 활동적이며 수용력이 강하고, 친절하며 낙천적이다. 어떤 상황이든 잘 적응하며 현실적이고 실질적이다. 주변에 관심이 많으며, 사람이나 사물을 다루는 사실적인 상식이 풍부하다. 그러나 이론적인 상식 면에선 놀라울 만큼 약할 수도 있다. 때로는 진지함이 결여되거나 마무리를 등한시하는 경향이 있으나, 조직이나 공동체에서 밝고 재미있는 분위기 조성 역할은 잘하는 편이다."
        elif mbti == "ENTJ": return "'타고난 리더'라고 불리는 이 유형은 권위와 자신감으로 역량을 발휘한다. 또한 공통된 목표를 통해 사람들을 끌어모으는 통솔력을 행사한다. 이들은 문제 해결 과정을 체계화하는 관리자/지도자가 자신의 역할이라고 생각한다. 주로 목표를 달성하기 위한 장기적인 계획 구상을 즐긴다. 분석적이고 객관적이며, 주변 세계에 자신만의 질서를 부여하는 것을 좋아한다. 질서의 결함이나 비효율성을 빠르게 파악하고, 더 나은 새로운 해결책을 찾는다. 또한 찾는 것에서 그치지 않고 기필코 현실에 구현해내려 하며, 그 과정을 즐긴다."
        elif mbti == "ENTP": return "특유의 능글거리면서 경쾌한 성격을 갖고 있다. 문제의 본질을 파악하고 논리적으로 판단하려는 기질이 있고, 어느 곳에서나 적응이 빠른 성격이다. 본인의 비전을 실현시키기 위해 노력하는 데다, 특유의 아웃사이더적인 성격까지 겹쳐 그야말로 혁명가의 기질을 띠고 있다. 모든 분야에 있어서, 기존의 체제 자체를 뒤집어 버리거나 전체의 도약을 이루어내는 인물들이 많다."
        elif mbti == "ENFJ": return "온화하고 적극적이며 책임감이 강하다. 사교성이 풍부하고 동정심이 많다. 상당히 이타적이고 민첩하고 사람 간의 화합을 중요시하며, 참을성이 많다. 다른 사람들의 생각이나 의견에 진지한 관심을 가지고, 대체로 동의한다. 미래의 가능성을 추구하며, 편안하고 능수능란하게 계획을 제시하고 집단을 이끌어가는 능력이 있다."
        elif mbti == "ENFP": return "정열적이고 활기가 넘치며 상상력이 풍부하다. 온정적이고 창의적이며, 항상 새로운 가능성을 찾고 시도하는 유형이다. 문제 해결에 재빠르고, 관심이 있는 일은 뭐든지 수행해내는 능력과 열성이 있다. 하지만 반복되는 일상적인 일은 참지 못하고 열정을 쏟지 않는다. 또한 한 가지 일을 끝내기도 전에 몇 가지 다른 일을 또 벌이는 경향을 가지고 있다."
        else: return "mbti 불명"

main_llm = MainLlm()