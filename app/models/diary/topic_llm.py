import json
from textwrap import dedent

from langchain_core.prompts import ChatPromptTemplate

from app.exception.exceptions import ControlledException, ErrorCode
from app.models.default_model import task_model

class TopicLlm:
    """
    요약:
        채팅 내역을 바탕으로 일기를 생성하는 Ollama 클래스

    설명:
        채팅 내역에서 주제를 추출하여 일기를 작성하는 모델이다.

    Attributes:
        __chain: llm을 활용하기 위한 lang_chain
    """
    def __init__(self):
        # 랭체인 생성
        prompt = ChatPromptTemplate.from_messages(self.__DIARY_TEMPLATE)
        self.__chain = prompt | task_model

    def topic_invoke(self, chat_rooms: list[str]) -> list[dict]:
        """
        요약:
            대화 내역에서 주제를 추출하여 일기를 생성하는 함수

        Parameters:
            chat_rooms(list[str]): 사용자의 대화 내역

        Raises:
            JSONDecodeError: JSON Decoding 실패 시, 작업 중단
            KeyError: 키 값에  "result"가 존재하지 않는 경우, 작업 중단
        """
        answer = self.__chain.invoke({"input": "\n".join(chat_rooms), "return_form_example":self.__RETURN_FORM_EXAMPLE, "result_example":self.__RESULT_EXAMPLE}).content

        try:
            diary = json.loads(answer)["result"]
        except json.JSONDecodeError:
            raise ControlledException(ErrorCode.FAILURE_JSON_PARSING)
        except KeyError:
            raise ControlledException(ErrorCode.INVALID_DATA_TYPE)
        return diary

    __DIARY_TEMPLATE = [
        ("system", "/no_think"),
        ("system", dedent("""
        <PRIMARY_RULE>
        무조건 JSON 형식을 유지해야 합니다.
        JSON 외에 자연어 해설은 없습니다.
        AI, 챗봇, 대화방, 시스템, 주석, 설명, 프롬프트 등 메타 표현은 절대 금지합니다.
        </PRIMARY_RULE>
        
        <ROLE>
        당신의 임무는 `Q.`에 있는 문장들로 일기를 작성하는 것입니다.
        일기는 1인칭 일기체를 사용합니다. (예: 나는 ~했다. 오늘은 ~였다.)
        </ROLE>
        
        <KNOWLEDGE>
        {input}
        </KONWLEDGE>
        
        <RULE>
        다음은 주어진 입력에 **필수적**으로 지켜야 할 반환 규칙입니다.
        - KNOWLEDGE는 일기에 이용될 대화 기록입니다.
        - KNOWLEDGE에 중복된 정보가 있다면 가장 최근의 문장을 이용합니다.
        - 제공된 대화기록외에 일기 작성에 **새로운 사실**은 절대 사용하면 안됩니다.
        </RULE>
        
        <EXCEPTION>
        만약 일기를 도출하지 못했다면, `empty list`(`[]`) 반환합니다.
        </EXCEPTION>
        
        <WRITING_INSTRUCTIONS>
        다음은 일기를 작성할 때, 지켜야 할 일기 작성 규칙입니다. 
        - 주제(`title`) 규칙
            - `title`은 **1문장 이내**로 핵심어를 이용해 작성해야 합니다.
        - 본문(`content`) 규칙
            - `content`는 3문장~10문장으로 완성해야 합니다.
            - 가능하면 감정이나 환경 묘사(시각/청각/후각)를 한 줄 이상 삽입해야 합니다.
        </WRITING_INSTRUCTIONS>
        
        <RETURN_TYPE>
        - 출력은 반드시 아래 예시와 동일한 JSON 구조로 반환합니다.
        - 최상위 `key`는 `result`입니다.
        - `result` `key`의 `value`는 `list[dict]` `type`입니다.
        - `list[dict]`에는 주제를 `dict type`으로 저장합니다.
        - `dict`의 `key`는 무조건 `title`, `content`만 가능합니다.
        </RETURN_TYPE>
        
        <RETURN_FORM>
        {return_form_example}
        </RETURN_FORM>
        
        <RESULT>
        {result_example}
        Q. <INPUT> {input} </INPUT>
        A. """))
    ]

    __RETURN_FORM_EXAMPLE = dedent("""
    {
        "result": [
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
    }
    """)

    __RESULT_EXAMPLE = dedent("""
    Q. <INPUT> U@Human: 안녕! 오늘 퇴근 후에 뭐 해? at 0000-00-00T00:00:00.000\nO@user_id_100: 별 계획 없는데? 영화나 볼까 생각 중이야. at 2025-05-20T18:01:25.901\nU@Human: 오! 그럼 ‘파묘’ 볼래? 다들 소름 돋는다고 하더라. at 2025-05-20T18:02:01.667\nO@user_id_100: 좋지! 7시 30분 홍대 CGV 예매해 둘게. at 2025-05-20T18:02:30.488\nU@Human: 끝나고 매운 라멘 먹자~ 요즘 날이 쌀쌀해서 딱일 듯. at 2025-05-20T18:03:04.210\nO@user_id_100: 콜! 영화관 앞에서 보자. at 2025-05-20T18:03:25.004 </INPUT>
    A. {"result": [{"title": "퇴근 후 영화·라멘 약속","content": "나는 퇴근 후 별다른 계획이 없어 친구와 영화를 보기로 했다. 우리는 홍대 CGV에서 저녁 7시 30분에 공포 영화 ‘파묘’를 예매했다. 쌀쌀한 밤공기 속에서 영화를 본 뒤 매운 라멘으로 몸을 데우기로 했는데, 매콤한 향을 상상하니 벌써 침이 고였다. 영화의 긴장감과 뜨거운 국물의 조합을 생각하니 기대감에 마음이 들떴다."}]}
    Q. <INPUT> U@Human: 오늘 팀장님이 기획안 싹 갈아엎으랬어… 멘탈 박살 😭 at 2025-05-20T12:15:33.882\nO@user_id_231: 헉 고생; 저녁에 헬스장 갈 건데 같이 털어버릴래? at 2025-05-20T12:16:11.402\nU@Human: 좋아! 러닝머신 뛰면서 욕 좀 해야겠다 ㅋㅋ at 2025-05-20T12:16:45.730\nO@user_id_231: 6시 반에 디지털미디어시티역 2번 출구 헬스장 고고! at 2025-05-20T12:17:10.957 </INPUT>
    A. {"result": [{"title": "기획안 스트레스, 헬스로 해소","content": "오늘 팀장님이 기획안을 통째로 다시 짜오라는 바람에 멘탈이 완전히 무너졌다. 머릿속이 새하얘질 정도로 짜증이 났지만, 친구의 제안 덕분에 저녁 6시 반에 디지털미디어시티역 헬스장에 가기로 했다. 러닝머신 위에서 땀을 쏟으며 속으로 쌓였던 욕을 마음껏 내뱉자 조금씩 기분이 풀렸다. 이어폰 속 비트가 심장과 맞춰 뛰자 온몸이 뜨거워지며 스트레스가 땀과 함께 흘러내리는 느낌이었다. 결국 운동이 끝나자 답답했던 가슴이 시원해졌고, 무너진 멘탈도 어느 정도는 다시 세워졌다."}]}
    Q. <INPUT> U@Human: 아침에 비 엄청 오더니 지금은 햇빛 쨍하네! at 2025-05-20T07:50:12.012\nO@user_id_045: 맞아, 우산 들고 나왔다가 민망😂 at 2025-05-20T07:51:08.620\nU@Human: 출근길에 무지개 봤어? 지하철 창밖에 살짝 떴던데. at 2025-05-20T07:51:40.300\nO@user_id_045: 못 봤어! 사진 찍었으면 공유해줘~ at 2025-05-20T07:52:05.477 </INPUT>
    A. {"result": [{"title": "변덕스러운 아침 하늘","content": "아침 출근길엔 장대비가 쏟아져 우산을 꼭 붙들고 나왔는데, 회사에 가까워질수록 구름이 걷히더니 금세 눈부신 햇빛이 퍼졌다. 젖은 도로 위에 햇살이 반짝여 눈이 시릴 정도였고, 지하철 창밖으로는 살짝 무지개가 떠서 잠깐이지만 마음이 설렜다. 갑작스러운 햇살 덕분에 무거웠던 우산이 순식간에 애물단지가 됐지만, 하늘이 준 작은 보상을 받은 기분이었다. 변덕스러운 날씨 덕에 하루를 시작하며 웃을 수 있었고, 괜히 상쾌한 기운이 몸에 번졌다."}]}
    Q. <INPUT> U@Human: 아침에 도착한 원두 택배 뜯어 봤어? 향이 엄청 진하대! at 2025-05-21T08:32:18.004\nO@user_id_150: 아직 못 뜯었어 ㅎㅎ 점심 때 핸드드립 내려서 같이 맛보자~ at 2025-05-21T08:33:02.741\nU@Human: 좋아, 브라질산이라는데 초콜릿 향 난다더라. at 2025-05-21T08:33:41.209\nO@user_id_150: 기대된다! at 2025-05-21T08:34:05.612\nU@Human: 근데 이번 주말 부산 기차표 예매했어? 자리 거의 다 찼다던데. at 2025-05-21T12:15:22.980\nO@user_id_150: 방금 KTX 11시 10분 표 잡았어! 해운대 근처 숙소도 예약 완료. at 2025-05-21T12:16:04.333\nU@Human: 굿! 토요일 밤엔 광안리 불꽃쇼도 한다더라. 스팟 좀 알아볼까? at 2025-05-21T12:16:48.100\nO@user_id_150: 오케이, 인스타에서 뷰 좋은 카페 찾으면 공유할게. at 2025-05-21T12:17:25.457\nU@Human: 참고로 금요일 프로젝트 데드라인 6시야. 오후 네 시엔 중간 점검 회의도 있고. at 2025-05-21T15:42:11.889\nO@user_id_150: 알지 알지! 오늘 안으로 KPI 표 다시 정리해서 드라이브에 올려 둘게. at 2025-05-21T15:42:55.220\nU@Human: 고마워~ 덕분에 내일 QA만 집중하면 될 듯. at 2025-05-21T15:43:28.571\nO@user_id_150: 파이팅! 끝나면 부산 여행 준비만 남았네 ㅎㅎ at 2025-05-21T15:44:02.004 </INPUT>
    A. {"result": [{"title": "신선한 원두 시향","content": "아침에 도착한 브라질산 원두를 열어 보니 봉투에서 달콤한 초콜릿 향이 확 퍼져 코끝이 깜짝 놀랐다. 점심에 핸드드립으로 내려 마시기로 약속하자 괜히 마음이 설렜다. 진한 향을 맡으며 오늘 하루가 기분 좋게 시작될 것 같아 가벼운 발걸음으로 일을 시작했다."},{"title": "부산 여행 준비","content": "점심시간에 주말 부산행 KTX 11시 10분 표를 예매했다는 소식을 듣고 가슴이 두근거렸다. 해운대 근처 숙소까지 예약을 마치니 여행이 눈앞에 그려졌다. 토요일 밤 광안리 불꽃쇼를 볼 생각에 벌써부터 귓가에 바닷소리가 들리는 듯했다. 인스타그램에서 바다 전망이 좋은 카페를 찾아볼 생각에 설렘이 더해졌다."},{"title": "프로젝트 마감 준비","content": "금요일 오후 여섯 시 데드라인을 앞두고 팀원들과 네 시 중간 점검 회의를 잡았다. 나는 오늘 안으로 KPI 표를 다시 정리해 드라이브에 올리기로 약속했다. 서류를 매만지며 키보드를 두드리는 소리가 사무실에 잔잔히 울릴 때, 일을 마치면 곧 부산으로 떠난다는 생각이 피로를 잊게 해 주었다."}]}
    """)

topic_llm = TopicLlm()