import os

import requests
from datetime import date

from dotenv import load_dotenv

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
            # else: name = chat["id"]
            else: name = chat["uid"]

            chat_room_log.append(f"{chat["type"]}@{name}: {chat["content"]} at {chat["chat_at"]}")

        user_all_chat_room_log.append(chat_room_log)
    return user_all_chat_room_log