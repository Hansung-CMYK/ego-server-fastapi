import json
import os
from datetime import date

import requests
from dotenv import load_dotenv

from config.external.common_function import parse_json

load_dotenv()

SPRING_URI = os.getenv('SPRING_URI')

"""
ego Table
"""
def get_ego(user_id:str):
    """
    요약:
        user_id로 본인의 ego_id를 조회하는 함수

    Parameters:
        user_id(str): 조회할 사용자의 아이디
    """
    url = f"{SPRING_URI}/api/v1/ego/user/{user_id}"
    response = requests.get(url)

    return parse_json(response, title="에고 조회 실패")

"""
personality Table
"""
def patch_tags(ego_id:str, tags:list[str]):
    """
    요약:
        사용자의 태그 정보를 업데이트 하기 위한 함수이다.

    Parameters:
        ego_id(str): 태그를 추가할 에고의 아이디
        tags(list[str]): 추가할 태그 목록
    """
    url = f"{SPRING_URI}/api/v1/ego"
    update_data = {"id": ego_id, "personalityList": tags}
    headers = {"Content-Type": "application/json"}
    response = requests.patch(url=url, data=json.dumps(update_data), headers=headers)

    return parse_json(response, title="태그 저장 실패")

"""
relationship Table
"""
def post_relationship(user_id:str, ego_id:str, relationship_id:int, target_date:date):
    """
    요약:
        당일 사용자-에고 관계를 추가하는 함수

    Parameters:
        user_id: 관계를 갖는 사용자 아이디
        ego_id: 관계를 갖는 에고 아이디
        relationship_id: 관계 아이디
        target_date: 관계가 이루어진 날짜
    """
    url = f"{SPRING_URI}/api/v1/ego-relationship"
    post_data = {"uid": user_id, "egoId": ego_id, "relationshipId": relationship_id,
                 "createdAt": target_date.isoformat()}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url=url, data=json.dumps(post_data, default=str), headers=headers)

    return parse_json(response, title="관계 저장 실패")