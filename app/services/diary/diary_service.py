import os
from datetime import date, datetime

from app.exception.exceptions import ControlledException, ErrorCode
from app.models.diary.preference_llm import preference_llm
from app.models.chat.persona_llm import persona_llm
from app.services.diary.tag_service import sentence_embedding, search_tags
from app.services.chatting.persona_store import persona_store
from app.models.database.postgres_database import postgres_database

import requests
from dotenv import load_dotenv
import json

load_dotenv()
SPRING_URI = os.getenv('SPRING_URI')

def get_all_chat(user_id: str, target_time: date):
    url = f"{SPRING_URI}/api/v1/chat-history/{user_id}/{target_time}"
    response = requests.get(url)
    chat_rooms = response.json()["data"]

    user_all_chat_room_log: list[list[str]] = []  # 사용자의 모든 채팅방 대화 목록
    for chat_room in chat_rooms:
        chat_room_log = []
        for chat in chat_room:
            if chat["type"] == "U": name = "Human"
            else: name = chat["id"]

            chat_room_log.append(f"{chat["type"]}@{name}: {chat["content"]} at {chat["chat_at"]}")

        user_all_chat_room_log.append(chat_room_log)
    return user_all_chat_room_log

async def async_save(user_id:str, all_chat:list[list[str]], target_date:date):
    """
    일기 작성에서 비동기 작업을 실행하는 함수이다.

    Parameters:
        user_id(str):
        all_chat(list[list[str]]):
        target_date(date):
    """
    stories = ["".join(chat_room) for chat_room in all_chat]

    # user_id로 my_ego 추출
    url = f"{SPRING_URI}/api/v1/my_ego/{user_id}/list"
    response = requests.get(url)
    my_ego = response.json()["data"][0]

    # 페르소나 저장
    save_persona(ego_id=my_ego["id"], stories=stories)

    # 태그 저장
    save_tags(ego_id=my_ego["id"], stories=stories)

    # 관계 저장
    for chat_room in all_chat:
        if chat_room[0][0] == "E": # 첫 채팅 타입이 E인 경우에만 작성(에고 채팅은 무조건 에고가 먼저)
            save_relation(user_id=user_id ,chat_room=chat_room, target_date=target_date)

    print("async_save success")
    return

def save_relation(user_id:str, chat_room:list[str], target_date:date):
    """
    BE에 관계정보를 추출해 저장하는 함수이다.

    Parameters:
        user_id(str): 사용자 본인의 ID
        chat_room(list[str]): 에고와 대화한 채팅 기록
        target_date(date): 대화를 기록할 날짜
    """
    ego_id = chat_room[0].split('@')[1].split(':')[0]

    relation = preference_llm.invoke(input=chat_room)
    # relationship_id = relationship_id_mapper(relation)
    relationship_id = 1

    # TODO: API 실행
    url = f"{SPRING_URI}/api/v1/ego-relationship"
    post_data = {"uid": user_id, "egoId": ego_id, "relationshipId": relationship_id, "createdAt": target_date.isoformat()}
    headers = {"Content-Type": "application/json"}

    response=requests.post(url=url, data=json.dumps(post_data, default=str), headers=headers)
    print(f"relation success: {response}")

def save_persona(ego_id:int, stories:list[str]):
    """
    BE에 페르소나 정보를 추출해 저장하는 함수이다.

    Parameters:
        ego_id(int): 본인의 에고 ID
        stories: 페르소나를 추출할 대화 내역
    """
    user_persona = persona_store.get_persona(ego_id=ego_id)
    delta_persona = persona_llm.invoke(
        current_persona=user_persona,
        input=stories
    )

    persona_store.update(ego_id=ego_id, delta_persona=delta_persona)  # 변경사항 업데이트

    postgres_database.update_persona(
        ego_id=ego_id,
        persona_json=persona_store.get_persona(ego_id=ego_id)
    )  # 데이터베이스에 업데이트 된 페르소나 저장
    print(f"delta_persona success: {delta_persona}")

def save_tags(ego_id:int, stories:list[str]):
    """
    BE에 에고 태그를 생성해 저장하는 함수이다.

    Parameters:
        ego_id(int): 본인의 에고 ID
        stories: 태그를 추출할 대화 내역
    """
    embedded_sentence = sentence_embedding(stories=stories)
    tags = search_tags(embedded_user_chat_logs=embedded_sentence)

    url = f"{SPRING_URI}/api/v1/ego"
    update_data = {"id": ego_id, "personalityList": tags}
    headers = {"Content-Type": "application/json"}

    response=requests.patch(url=url, data=json.dumps(update_data), headers=headers)
    print(f"tags success: {response}")

def relationship_id_mapper(relation:str):
    """
    매력적, 즐거운, 만족한, 원만한, 지루한, 불안한, 부정적의 ID를 매핑해서 반환해줍니다.
    """
    if relation == "매력적": return 1
    elif relation == "즐거운": return 2
    elif relation == "만족한": return 3
    elif relation == "원만한": return 4
    elif relation == "지루한": return 5
    elif relation == "불안한": return 6
    elif relation == "부정적": return 7
    else: raise ControlledException(ErrorCode.INVALID_RELATIONSHIP)