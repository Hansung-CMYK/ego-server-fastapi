from fastapi import APIRouter
from pydantic import BaseModel
import asyncio

from app.api.common_response import CommonResponse
from app.exception.exceptions import ControlledException, ErrorCode
from app.models.diary.daily_comment_llm import daily_comment_llm
from app.models.diary.topic_llm import topic_llm
from app.models.diary.keyword_model import keyword_model
from datetime import date
from app.services.diary.diary_service import async_save, get_all_chat

from app.services.diary.kobert_handler import extract_emotions
router = APIRouter()

class DiaryRequest(BaseModel):
    user_id: str
    ego_id: int
    target_date: date

@router.post("/diary")
async def to_diary(body: DiaryRequest):
    # NOTE 1. SQL 조회로 이전하기
    all_chat = get_all_chat(user_id=body.user_id, target_time=body.target_date)

    stories = ["".join(chat_room) for chat_room in all_chat]

    # NOTE 2. 키워드 추출
    keywords = keyword_model.get_keywords(stories=stories, count=5)

    # NOTE 3. 일기 생성
    # 예외처리: 일기 생성 전, 일기를 생성하기 위한 문장 수가 충분한지 확인
    if sum(len(story) for story in stories) < 5: # 리스트에 있는 전체 문장 수가 5개 이상이어야 한다.
        raise ControlledException(ErrorCode.CHAT_COUNT_NOT_ENOUGH)

    # 일기 생성
    topics = topic_llm.invoke(story=stories)

    # 예외처리: 일기로 아무 내용이 반환되지 않았는지 확인한다.
    if len(topics) < 1: raise ControlledException(ErrorCode.CAN_NOT_EXTRACT_DIARY)

    # NOTE 4. 감정 분석
    feeling = extract_emotions(topics)

    # NOTE 5. 한줄 요약 문장 생성
    daily_comment = daily_comment_llm.invoke(diaries=topics, feelings=feeling, keywords=keywords)

    # NOTE 6. 에고 페르소나 수정
    asyncio.create_task(async_save(user_id=body.user_id, all_chat=all_chat, target_date=body.target_date))

    # NOTE 7. FE 반환 diary 객체 생성
    # TODO: 문장 변경 가능성 있음.
    diary = {
        "uid": body.user_id,
        "egoId": body.ego_id,
        "feeling": ",".join(feeling),
        "dailyComment": daily_comment,
        "createdAt": body.target_date,
        "keywords": keywords,
        "topics":  topics
    }

    return CommonResponse(
        code=200,
        message="diary success!",
        data=diary
    )