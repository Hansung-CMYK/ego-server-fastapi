from app.routers.persona import persona_repository


def create_persona():
    persona_repository.create_persona()

def drop_persona():
    persona_repository.drop_persona()

def has_persona()->bool:
    return persona_repository.has_persona()

def select_persona_to_ego_id(ego_id: str)->tuple:
    """
    요약:
        ego_id를 이용해 페르소나를 조회하는 함수

    Parameters:
        ego_id(str): 조회할 ego의 아이디

    Raises:
        PERSONA_NOT_FOUND: ego_id로 페르소나 조회 실패
    """
    return persona_repository.select_persona_to_ego_id(ego_id=ego_id)

def insert_persona(ego_id: str, persona: dict):
    """
    요약:
        페르소나를 추가하는 함수

    Parameters:
        ego_id(str): 추가할 에고의 아이디 * BE ego 테이블과 1대1 매핑되어야 한다.
        persona(dict): 추가할 페르소나 정보
    """
    return persona_repository.insert_persona(ego_id=ego_id, persona=persona)

def update_persona(ego_id: str, persona: dict):
    """
    요약:
        기존 페르소나를 변경하는 함수

    Parameters:
        ego_id(str): 변경할 에고의 아이디
        persona(dict): 새로 저장할 사용자의 페르소나\
    """
    return persona_repository.update_persona(ego_id=ego_id, persona=persona)

def delete_persona(ego_id: str):
    """
    모든 데이터를 제거하는 함수
    """
    return persona_repository.delete_persona(ego_id=ego_id)

def has_persona(ego_id: str) -> bool:
    """
    요약:
        persona 테이블에 이미 ego_id가 존재하는지 확인하는 함수

    Parameters:
        ego_id(str): 존재하는지 확인힐 ego 아이디
    """
    return persona_repository.has_persona(ego_id=ego_id)

def persona_filter(request_body:dict) -> dict:
    """
    request_body에서 페르소나 데이터만 필터링 해주는 함수
    """
    request_body.pop("ego_id")
    request_body.pop("interview")
    return {key: value for key, value in request_body.items() if value is not None}