import json
from textwrap import dedent

from langchain_core.prompts import ChatPromptTemplate

from app.exception.exceptions import ControlledException, ErrorCode
from app.models.default_model import task_model, DEFAULT_TASK_LLM_TEMPLATE, clean_json_string, llm_sem
from app.logger.logger import logger

class TopicLlm:
    def __init__(self):
        prompt = ChatPromptTemplate.from_messages(self.__DIARY_TEMPLATE)
        self.__chain = prompt | task_model

    def topic_invoke(self, summary: str) -> list[dict]:
        with llm_sem:
            answer = self.__chain.invoke({
                "input": summary,
                "return_form_example": self.__RETURN_FORM_EXAMPLE,
                "result_example": self.__RESULT_EXAMPLE,
                "default_task_llm_template": DEFAULT_TASK_LLM_TEMPLATE,
            }).content
        clean_answer: str = clean_json_string(text=answer)

        logger.info(msg=f"\n\nPOST: api/v1/diary [일기 생성 LLM 결과]\n{clean_answer}\n")

        try:
            return json.loads(clean_answer)["result"]
        except json.JSONDecodeError:
            raise ControlledException(ErrorCode.FAILURE_JSON_PARSING)
        except KeyError:
            raise ControlledException(ErrorCode.INVALID_DATA_TYPE)

    __DIARY_TEMPLATE = [
        ("system", """
        /no_think
        {default_task_llm_template}
        """),
        ("system", dedent("""
        <PRIMARY_RULE>
        1. Return ONLY valid JSON – no additional text.
        2. No comments, markdown, or system tags.
        3. Use straight double-quotes (").
        4. All output must be in fluent, natural Korean.
        </PRIMARY_RULE>

        <ROLE>
        * You are a sensitive and precise diary writer.
        * Turn the summarized chat log into reflective diary entries.
        </ROLE>

        <KNOWLEDGE>
        * Use only lines starting with `u@` as diary content.
        * Treat `e@` lines as reference context — do not quote or summarize them.
        * If there is no usable content → respond with exactly [] as "result".
        </KNOWLEDGE>

        <WRITING_GUIDELINES>
        * Split the day into multiple diary topics, each as an object.
        * Each object:
          - "title": 1 short sentence (clear topic).
          - "content": 3–10 fluent sentences with at least one emotional or sensory element.
        * Do not hallucinate or infer beyond the summarized input.
        * Keep the total output within 1,000 tokens.
        </WRITING_GUIDELINES>

        <OUTPUT_SCHEMA>
        {return_form_example}
        </OUTPUT_SCHEMA>

        <RETURN_EXAMPLE>
        {result_example}
        </RETURN_EXAMPLE>

        Q. <INPUT> {input} </INPUT>
        A.
        """))
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