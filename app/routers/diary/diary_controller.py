import asyncio
import logging

from fastapi import APIRouter

from app.internal.exception.error_code import ControlledException, ErrorCode
from app.routers.diary.diary_service import async_save, get_all_chat
from app.routers.diary.dto.diary_request import DiaryRequest
from app.routers.diary.feeling.kobert_handler import extract_emotions
from config.common.common_response import CommonResponse
from config.keyword.keyword_model import keyword_model
from config.llm.daily_comment_llm import DailyCommentLLM
from config.llm.summary_llm import SummaryLLM
from config.llm.topic_llm import TopicLLM

router = APIRouter(prefix="/diary")


@router.post("")
async def create_diary(body: DiaryRequest)->CommonResponse:
    """
    요약:
        대화 내역을 기반으로 일기를 작성하는 API

    설명:
        - 일기 생성과 함께 몇가지 작업이 함께 진행된다.

        - 키워드 추출: 대화 내역에서 중요도가 높은 키워드(단어) 추출
        - 일기 생성: 대화 내역으로 일기를 생성
        - 감정 분석: 대화 내역에서 발생한 감성 분석
        - 한줄 요약 문장 생성: 키워드, 일기, 감정을 기반으로 한줄평 생성
        - 에고 페르소나 수정: 대화내역을 바탕으로 본인 에고의 페르소나를 수정

    Parameters:
        body(DiaryRequest): 일기 생성에 필요한 인자의 모음
            * user_id,ego_id, target_date를 Attribute로 갖는다.
    """
    # NOTE 1. 대화 기록 조회 후, 데이터 가공
    # taget_date 기준 24시간 이내 채팅을 불러온다.
    # all_chat: 모든 채팅방의 채팅이 list[list[str]]로 저장된 변수이다.
    all_chat:list[list[str]] = get_all_chat(user_id=body.user_id, target_date=body.target_date)

    # chat_rooms: 채팅방 별 채팅이 str로 묶어서 list[str]로 저장된 변수이다.
    chat_rooms:list[str] = ["\n".join(chat_room) for chat_room in all_chat] # 채팅방 별로 대화 내역을 통합한다.

    # LOG. 시연용 로그
    logging.info(msg=f"\n\nPOST: api/v1/diary [채팅방 대화기록]\n{chat_rooms[0]}\n")

    # NOTE 2. 키워드 추출
    keywords:list[str] = keyword_model.get_keywords(chat_rooms=chat_rooms) # 최대 5개의 키워드를 추출한다.

    # NOTE 3. 일기 생성
    # TODO 1. 서비스 시연을 위해 일기 생성 문장 수를 5개로 감축하였다.
    # 예외처리 1. 일기 생성 전, 일기를 생성하기 위한 문장 수가 충분한지 확인한다.
    if sum(len(all_chat_of_chat_room) for all_chat_of_chat_room in all_chat) < 5: # 24시간 내에 대화한 채팅이 5개 이상이어야 한다.
        raise ControlledException(ErrorCode.CHAT_COUNT_NOT_ENOUGH)

    # 대화 내역 요약
    summary = SummaryLLM().invoke({
        "input": "\n".join(chat_rooms)
    })

    # 일기 생성
    # 각 주제가 list[{"title": str, "content": str}]로 저장된다.
    topics:list[dict] = TopicLLM().invoke({"input": summary})

    # 예외처리 2. 일기로 아무 내용이 반환되지 않았는지 확인한다.
    if len(topics) < 1:
        raise ControlledException(ErrorCode.CAN_NOT_EXTRACT_DIARY)

    # NOTE 4. 감정 분석
    # 일기 내용을 바탕으로 감정을 추출한다.
    contents = [topic["content"] for topic in topics]
    feeling:list[str] = extract_emotions(contents)

    # NOTE 5. 한줄 요약 문장 생성
    # 키워드, 일기, 감정을 기반으로 한줄평을 생성한다.
    events = [topic["title"] for topic in topics]  # 일기에서 제목(주제)를 정제한다.
    daily_comment:str = DailyCommentLLM().invoke({
        "events": events, "feelings": feeling, "keywords": keywords
    })

    # NOTE 6. 에고 페르소나 수정
    # 일기 생성과 동시에 이루어져야 할 작업을 수행한다.
    asyncio.create_task(async_save(user_id=body.user_id, chat_rooms=chat_rooms, target_date=body.target_date))

    # NOTE 7. FE 반환 diary 객체 생성
    diary:dict = {
        "uid": body.user_id,
        "egoId": body.ego_id,
        "feeling": ",".join(feeling), # FE, BE 요청 사항으로 감정 리스트를 문자열로 변경하여 전달
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