from app.internal.admin.dto.admin_request import ADMIN_ID, ADMIN_PASSWORD
from config.external import hub_api


def check_authorization(admin_id:str, admin_password:str)->bool:
    return admin_id == ADMIN_ID and admin_password == ADMIN_PASSWORD

def check_correct_user(user_id:str, ego_id:str)->bool:
    """
    user_id와 ego_id가 같은 사용자의 것인지 확인하는 함수
    """
    return hub_api.get_ego(user_id=user_id) == int(ego_id)