import os
from datetime import date

from app.exception.exceptions import ControlledException, ErrorCode
from app.models.chat.persona_llm import persona_llm
from app.models.emotion.emotion_classifier import EmotionClassifier
from app.services.api_service import get_chat_history, get_ego, post_relationship, patch_tags
from app.services.chat.persona_store import persona_store
from app.models.database.postgres_database import postgres_database
from app.logger.logger import logger

from dotenv import load_dotenv

from app.services.diary.tag_service import search_tags

load_dotenv()
SPRING_URI = os.getenv('SPRING_URI')

def get_all_chat(user_id: str, target_date: date):
    """
    요약:
        사용자가 하루동안 한 채팅 내역을 문자열로 정제하는 함수

    Parameters:
        user_id(str): 조회할 사용자 아이디
        target_date(date): 검색할 날짜
    """
    # NOTE 1. 대화 내역 조회
    chat_rooms = get_chat_history(user_id=user_id, target_date=target_date)

    # NOTE 2. 대화 내역 정제
    all_chat: list[list[str]] = []  # 사용자의 모든 채팅방 대화 목록
    for chat_room in chat_rooms:
        chat_room_log = []
        for chat in chat_room:
            if chat["type"] == "U": name = "Human"
            else: name = chat["id"]

            chat_room_log.append(f"{chat["type"]}@{name}: {chat["content"]} at {chat["chat_at"]}")

        all_chat.append(chat_room_log)
    return all_chat

async def async_save(user_id:str, chat_rooms:list[str], target_date:date):
    """
    요약:
        일기 작성에서 비동기 작업을 실행하는 함수

    설명:
        * 일기 생성과 동시에 수행되어야 할 각종 작업을 수행한다.
        * 관계 생성: 대화 내역을 기반으로 당일 에고와 사용자 간의 관계를 생성한다.
        * 페르소나 변경: 대화 내역을 기반으로 페르소나를 재설정한다.
        * 태그 생성: 대화 내역을 기반으로 당일 태그를 생성한다.

    Parameters:
        user_id(str): 저장할 사용자의 아이디
        chat_rooms(list[list[str]]): 사용자의 채팅 내역
        target_date(date): 저장할 날짜
    """
    # NOTE 1. user_id로 my_ego 추출
    my_ego = get_ego(user_id=user_id)
    ego_id:str = str(my_ego["id"]) # ego_id 문자열 변환

    # NOTE 2. 페르소나 저장
    save_persona(ego_id=ego_id, chat_rooms=chat_rooms)

    # NOTE 3. 태그 저장
    save_tags(ego_id=ego_id, stories=chat_rooms)

    # NOTE 4. 관계 저장
    for chat_rooms in chat_rooms: # 관계는 채팅방 별로 저장되어야 한다.
        if chat_rooms[0][0] == "E": # 첫 채팅 타입이 E인 경우에만 작성(에고 채팅은 무조건 에고가 먼저)
            save_relation(user_id=user_id, chat_room=chat_rooms, target_date=target_date)
    logger.info("async_save success!")

def save_relation(user_id:str, chat_room:str, target_date:date):
    """
    요약:
        BE에 관계정보를 추출해 저장하는 함수

    Parameters:
        user_id(str): 사용자 본인의 ID
        chat_room(list[str]): 에고와 대화한 채팅 기록
        target_date(date): 대화를 기록할 날짜
    """
    ego_id = chat_room.split('@')[1].split(':')[0]
    # [('억울함', 0.8254460692405701)]
    relation = EmotionClassifier().predict(texts="\n".join(chat_room))
    relationship_id = EmotionClassifier().mapper(relation=relation[0][0])

    # LOG. 시연용 로그
    logger.info(msg=f"\n\nPOST: api/v1/diary [에고 관계]\n{relation}\n""")

    post_relationship(user_id=user_id, ego_id=ego_id, relationship_id=relationship_id, target_date=target_date)
    logger.info("save_relation success!")

def save_persona(ego_id:str, chat_rooms:list[str]):
    """
    요약:
        BE에 페르소나 정보를 추출해 저장하는 함수

    Parameters:
        ego_id(str): 본인의 에고 ID
        chat_rooms: 페르소나를 추출할 대화 내역
    """
    # NOTE 1. 사용자 본인 페르소나를 조회한다.
    user_persona = persona_store.get_persona(ego_id=ego_id)

    # NOTE 2. 대화내역을 바탕으로 변경사항을 추출한다.
    delta_persona = persona_llm.persona_invoke(
        user_persona=user_persona,
        session_history=chat_rooms
    )

    # NOTE 3. 변경사항을 저장한다.
    persona_store.update(ego_id=ego_id, delta_persona=delta_persona)  # 변경사항 업데이트

    # NOTE 4. 변경사항을 DB에 저장한다.
    postgres_database.update_persona(
        ego_id=ego_id,
        persona=persona_store.get_persona(ego_id=ego_id)
    )  # 데이터베이스에 업데이트 된 페르소나 저장

    # NOTE 5. 메모리에서 자신의 페르소나를 내린다.
    # 메모리 과사용 방지를 위한 작업
    persona_store.remove_persona(ego_id=ego_id)
    logger.info("save_persona success!")

def save_tags(ego_id:str, stories:list[str]):
    """
    요약:
        BE에 에고 태그를 생성해 저장하는 함수

    Parameters:
        ego_id(str): 본인의 에고 ID
        stories(list[str]): 태그를 추출할 대화 내역
    """
    # NOTE 1. 대화 내역과 높은 유사도를 가진 태그를 조회한다.
    tags = search_tags(stories=stories)

    # LOG. 시연용 로그
    logger.info(msg=f"\n\nPOST: api/v1/diary [에고 태그]\n{tags}\n")

    # NOTE 2. 추출된 태그를 BE로 전달한다.
    patch_tags(ego_id=ego_id, tags=tags)
    logger.info("save_tags success!")

