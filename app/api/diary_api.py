from fastapi import APIRouter
from pydantic import BaseModel

from app.api.common_response import CommonResponse
from app.models.diary_llm_model import diary_llm
from datetime import date

router = APIRouter()

class DiaryRequest(BaseModel):
    message: str
    user_id: str
    ego_id: str

@router.post("/diary")
async def to_diary(body: DiaryRequest):
    # TODO: SQL 조회로 이전하기
    # stories = postgres_client.search_all_chat(user_id=body.user_id)
    stories = [
        [  # 채팅방 1.
            "AI: 안녕 오늘은 무슨 일이 있었어? at 2025년 05월 16일 05시 34분",
            "Human: 나는 오늘 고급 레스토랑에서 밥을 먹었어 at 2025년 05월 16일 05시 35분",
            "AI: 고급 레스토랑이라니 부럽다! 거기서 무슨 밥을 먹었는데? at 2025년 05월 16일 05시 35분",
            "Human: 안심 스테이크랑 트러플 파스타를 먹었어 at 2025년 05월 16일 05시 36분",
            "AI: 와 정말 맛있었겠다! 분위기도 좋았어? at 2025년 05월 16일 05시 36분",
            "Human: 응, 조용하고 조명이 은은해서 편하게 식사할 수 있었어 at 2025년 05월 16일 05시 37분",
            "AI: 그런 곳에서 식사하면 하루 스트레스가 다 날아가겠다 :) at 2025년 05월 16일 05시 37분"
        ],
        [  # 채팅방 2.
            "AI: 이번 주말에 뭐 할 계획이야? at 2025년 05월 16일 11시 12분",
            "Human: 친구들이랑 제주도로 여행 가기로 했어 at 2025년 05월 16일 11시 13분",
            "AI: 와 좋겠다! 제주도에서 어디 갈 예정이야? at 2025년 05월 16일 11시 13분",
            "Human: 우도랑 협재 해변, 그리고 한라산 등반도 계획하고 있어 at 2025년 05월 16일 11시 14분",
            "AI: 엄청 알찬 일정이네! 숙소는 어디 잡았어? at 2025년 05월 16일 11시 14분",
            "Human: 애월 쪽 뷰 좋은 숙소로 예약했어 at 2025년 05월 16일 11시 15분",
            "AI: 애월은 노을도 예쁘고 분위기도 좋더라! 재밌게 다녀와 :) at 2025년 05월 16일 11시 15분",
            "Human: 근데 어제 우도에서 다른 여행 온 친구랑 친해졌어 at 2025년 05월 16일 18시 22분",
            "AI: 진짜? 어떤 친구야? 여행지에서 만난 인연은 오래 기억에 남더라 at 2025년 05월 16일 18시 23분",
            "Human: 나랑 또래였고 혼자 여행 온 사람이었는데 말도 잘 통하고 사진도 같이 찍었어 at 2025년 05월 16일 18시 24분",
            "AI: 우와 분위기 좋았겠다! 연락처도 교환했어? at 2025년 05월 16일 18시 24분",
            "Human: 응, 인스타 교환했는데 다음에 서울 오면 또 보기로 했어 at 2025년 05월 16일 18시 25분",
            "AI: 여행에서 그렇게 친구 생기면 진짜 좋지! 좋은 인연으로 이어졌으면 좋겠다 :) at 2025년 05월 16일 18시 25분"
        ]
    ]

    # TODO: 감정 분석
    feeling = "샘플 감정1, 샘플 감정2"

    # TODO: 한줄평 요약
    daily_comment = "샘플 한줄평 요약"

    # TODO: 키워드 추출
    keyword = ["샘플1", "샘플2"]

    chat_count = sum(len(story) for story in stories)
    if chat_count < 5:
        raise Exception("-1: 일기를 만들기 위한 대화 수가 부족합니다.")

    topics = diary_llm.diary_invoke(story=stories)
    if len(topics) < 1:
        raise Exception("-3: 아무런 주제도 도출되지 못했습니다.")

    for topic in topics:
        content = topic["content"]
        # TODO: 이미지 저장하기
        topic.update({"url": f"TODO {content}"})

    # TODO: 일기 저장 API 전송
    print({
        "uid": body.user_id,
        "egoId": 1,
        "feeling": feeling,
        "dailyComment": daily_comment,
        "createdAt": date.today(),
        "keywords": keyword,
        "topics": topics,
    })

    return CommonResponse(
        code=200,
        message="diary success!"
    )