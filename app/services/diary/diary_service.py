import os
from datetime import date, datetime

from app.exception.exceptions import ControlledException, ErrorCode
from app.models.chat.persona_llm import persona_llm
from app.models.emotion.emtion_classifier import EmotionClassifier
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
    url = f"{SPRING_URI}/api/v1/ego/user/{user_id}"
    response = requests.get(url)
    my_ego = response.json()["data"]

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

    relation = EmotionClassifier().predict(texts="\n".join(chat_room))
    relationship_id = relationship_id_mapper(relation)

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
        session_history=stories
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

def relationship_id_mapper(relation: str):
    """
    각 감적의 BE.relationship ID를 매핑해서 반환해줍니다.
    """
    if relation == "admiration": return 1      # 감탄, 존경
    elif relation == "amusement": return 2     # 즐거움, 재미
    elif relation == "anger": return 3         # 분노
    elif relation == "annoyance": return 4     # 짜증
    elif relation == "approval": return 5      # 승인, 호의
    elif relation == "caring": return 6        # 보살핌
    elif relation == "confusion": return 7     # 혼란
    elif relation == "curiosity": return 8     # 호기심
    elif relation == "desire": return 9        # 욕망, 바람
    elif relation == "disappointment": return 10  # 실망
    elif relation == "disapproval": return 11  # 반감, 비판
    elif relation == "disgust": return 12      # 혐오
    elif relation == "embarrassment": return 13  # 당황, 민망
    elif relation == "excitement": return 14   # 흥분, 들뜸
    elif relation == "fear": return 15         # 두려움
    elif relation == "gratitude": return 16    # 감사
    elif relation == "grief": return 17        # 슬픔, 비탄
    elif relation == "joy": return 18          # 기쁨
    elif relation == "love": return 19         # 사랑
    elif relation == "nervousness": return 20  # 긴장
    elif relation == "optimism": return 21     # 낙관
    elif relation == "pride": return 22        # 자부심
    elif relation == "realization": return 23  # 깨달음
    elif relation == "relief": return 24       # 안도
    elif relation == "remorse": return 25      # 후회
    elif relation == "sadness": return 26      # 슬픔
    elif relation == "surprise": return 27     # 놀람
    elif relation == "neutral": return 28      # 중립
    else: raise ControlledException(ErrorCode.INVALID_RELATIONSHIP)
