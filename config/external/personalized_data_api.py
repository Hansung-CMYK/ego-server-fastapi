import os
from datetime import date

import requests
from dotenv import load_dotenv

from config.external.common_function import parse_json

load_dotenv()

SPRING_URI = os.getenv('SPRING_URI')

def get_chat_history(user_id:str, target_date:date):
    """
    요약:
        사용자가 하루동안 한 채팅 내역을 불러오는 함수

    Parameters:
        user_id(str): 조회할 사용자 아이디
        target_date(date): 검색할 날짜
    """
    url = f"{SPRING_URI}/api/v1/chat-history/{user_id}/{target_date}"
    response = requests.get(url)

    return parse_json(response, title="채팅 기록 조회 실패")