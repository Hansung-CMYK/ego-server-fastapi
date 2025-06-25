from app.routers.tone import tone_repository


def interview_to_str(interview: list[list[str]]) -> str:
    """
    인터뷰(2중 리스트)에서 첫 두 리스트를 교대로 이어붙여
    Java StringBuilder 로직과 동일한 로그 문자열을 반환합니다.

    Parameters
    ----------
    interview : list[list[str]]
        [[Q1, Q2, …], [A1, A2, …], …] 형태의 중첩 리스트

    Returns
    -------
    str
        Q1\nA1\nQ2\nA2\n … 형식의 문자열
    """
    if len(interview) < 2:           # 리스트가 2개 미만이면 빈 문자열 반환
        return ""

    first_list, second_list = interview[0], interview[1]

    lines: list[str] = []
    for q, a in zip(first_list, second_list):
        lines.append(q)
        lines.append(a)

    return "\n".join(lines)

def select_tone_to_ego_id(ego_id: str)->tuple:
    return tone_repository.select_tone_to_ego_id(ego_id=ego_id)

def insert_tone(ego_id: str, tone: dict):
    """
    요약:
        말투를 추가하는 함수

    Parameters:
        ego_id(str): 추가할 에고의 아이디 * BE ego 테이블과 1대1 매핑되어야 한다.
        tone(dict): 추가할 말투 정보
    """
    return tone_repository.insert_tone(ego_id=ego_id, tone=tone)

def update_tone(ego_id: str, tone:dict):
    return tone_repository.update_tone(ego_id=ego_id, tone=tone)

def delete_tone(ego_id: str):
    """
    모든 데이터를 제거하는 함수

    Parameters:
        ego_id: 제거할 ego_id
    """
    return tone_repository.delete_tone(ego_id=ego_id)

def has_tone(ego_id: str) -> bool:
    """
    요약:
        tone 테이블에 이미 ego_id가 존재하는지 확인하는 함수

    Parameters:
        ego_id(str): 존재하는지 확인힐 ego 아이디
    """
    return tone_repository.has_tone(ego_id=ego_id)