from langchain_core.prompts import ChatPromptTemplate

from app.models.default_model import chat_model

class PreferenceLlm:
    def __init__(self):
        # 메인 모델 프롬프트 적용 + 랭체인 생성
        prompt = ChatPromptTemplate.from_messages(self.__PREFERENCE_TEMPLATE)

        self.__preference_chain = prompt | chat_model

    def invoke(self, input: list[str]) -> str:
        """
        채팅방 대화내역을 바탕으로 사용자가 느낄 감정을 표현한다.
        """
        text = "\n".join(input)
        return self.__preference_chain.invoke(
            {"input": text}
        ).content

    __PREFERENCE_TEMPLATE = [
        ("system", "/no_think\n"),
        (
            "system", """
            <OBJECTIVE>
            • 입력된 대화(또는 문장 집합)를 읽고, Human이 상대방에게 느낀 관계/호감 수준을
              **관계 키워드 리스트** 중 가장 적절한 **하나**로 분류한다.
            </OBJECTIVE>
            
            <관계 키워드 & 선택 기준>
            1. 매력적   : 설렘·강한 호감·적극적 관심·칭찬
            2. 즐거운   : 유쾌·즐거움·웃음·긍정적 에너지
            3. 만족한   : 괜찮음·고마움·대체로 긍정 (큰 설렘은 아님)
            4. 원만한   : 중립·무난·예의적 교류(특별 감정 미미)
            5. 지루한   : 따분·흥미 부족·권태
            6. 불만족   : 아쉬움·실망·불평·개선 요구(공격적 언사·분노 표현 없음)
            
            <OUTPUT RULES>
            • 반드시 **한 단어**(위 키워드 그대로) 또는 `"분석 실패"` 만 출력  
              - 예: `즐거운` / `분석 실패`
            • 마침표·따옴표·접두어(“키워드: ” 등)·해설 **절대 금지**
            • 복합 감정이 섞여 있으면 **리스트 상에서 더 앞쪽(긍정적) 키워드**를 선택
            """
        ),
        ("human", "{input}"),
    ]

preference_llm= PreferenceLlm()