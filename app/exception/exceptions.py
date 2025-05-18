from enum import Enum

class ErrorCode(Enum):
    """
    예측 가능한 에러를 상수로 명시해둔 ErrorCode 클래스
    """
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message

    # 음성&채팅 메세지
    PASSAGE_NOT_FOUND = (-100, "해당 passage_id로 조회된 본문(passage)이 없습니다.")
    PERSONA_NOT_FOUND = (-101, "해당 ego_id로 조회된 페르소나(persona)가 없습니다.")
    # 일기 생성
    CHAT_COUNT_NOT_ENOUGH = (-201, "일기를 만들기 위한 문장 수가 부족합니다.")
    CAN_NOT_EXTRACT_DIARY = (-202, "아무런 주제도 도출되지 못했습니다.")
    FAILURE_JSON_PARSING = (-203, "일기 생성 중 JSON 변환이 실패되었습니다.")
    INVALID_DATA_TYPE = (-204, "LLM이 잘못된 데이터 타입을 생성했습니다.")
    POSTGRES_ACCESS_DENIED = (-205, "PostgreSQL 접속(or 스키마 접속)에 실패하였습니다.")
    INVALID_SQL_ERROR = (-206, "잘못된 SQL로 에러가 발생했습니다.")


class ControlledException(RuntimeError):
    """
    예측 가능한 에러를 관리하기 위한 Exception Class
    """
    def __init__(self, error_code: ErrorCode):
        self.error_code = error_code