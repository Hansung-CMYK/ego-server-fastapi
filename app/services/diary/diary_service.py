import os

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

async def async_save(user_id:str, all_chat:list[list[str]]):
    stories = ["".join(chat_room) for chat_room in all_chat]

    # TODO: user_id로 ego_id 추출하기
    ego_id = 1

    """
    일기 작성에서 비동기 작업을 실행하는 함수이다.
    """
    save_persona(ego_id=ego_id, stories=stories)
    for chat_room in all_chat:
        if chat_room[0][0] == "E": # 첫 채팅 타입이 E인 경우에만 작성(에고 채팅은 무조건 에고가 먼저)
            save_relation(chat_room)
    save_tags(ego_id=ego_id, stories=stories)
    return

def save_relation(chat_room:list[str]):
    """
    BE에 관계정보를 추출해 저장하는 함수이다.
    """
    ego_id = chat_room[0].split('@')[1].split(':')[0]

    relation = preference_llm.invoke(input=chat_room)

    # TODO: API 실행
    return

def save_persona(ego_id:int, stories:list[str]):
    """
    BE에 페르소나 정보를 추출해 저장하는 함수이다.
    """
    user_persona = persona_store.get_persona(ego_id=ego_id)
    delta_persona = persona_llm.invoke(
        current_persona=user_persona,
        session_history=stories
    )
    print(f"delta_persona: {delta_persona}")
    persona_store.update(persona_id=ego_id, delta_persona=delta_persona)  # 변경사항 업데이트

    postgres_database.update_persona(
        persona_id=ego_id,
        persona_json=persona_store.get_persona(persona_id=ego_id)
    )  # 데이터베이스에 업데이트 된 페르소나 저장

def save_tags(ego_id:int, stories:list[str]):
    """
    BE에 에고 태그를 생성해 저장하는 함수이다.
    """
    embedded_sentence = sentence_embedding(stories=stories)
    tags = search_tags(embedded_user_chat_logs=embedded_sentence)

    url = f"{SPRING_URI}/api/v1/ego"
    update_data = {"id": ego_id, "personalityList": tags}
    headers = {"Content-Type": "application/json"}

    requests.patch(url=url, data=json.dumps(update_data), headers=headers)
