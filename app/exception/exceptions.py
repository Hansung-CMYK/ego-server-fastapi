from enum import Enum

class ErrorCode(Enum):
    """
    요약:
        예측 가능한 에러를 상수로 명시해둔 ErrorCode 클래스

    설명:
        각 반환 코드는 음수로 표시된다.
        첫번째 값은 에러들의 항목을 의미한다.

    Attributes:
        code(int): 에러코드
        message(str): 에러 메세지
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
    # 관계 생성
    INVALID_RELATIONSHIP = (-301, "잘못된 관계가 도출되었습니다. *모델 문제")
    # 에고 생성
    ALREADY_CREATED_EGO_ID = (-401, "Ego 정보가 이미 Postgres Database에 존재합니다.")
    ALREADY_CREATED_PARTITION = (-402, "Ego 정보가 이미 Milvus Database에 존재합니다.")
    PARTITION_NOT_FOUND = (-403, "존재하지 않는 Milvus Partition입니다.")
    # 이미지 생성 에러
    IMAGE_DESCRIPTION_ERROR = (-501, "이미지를 분석하는 과정에서 오류가 발생했습니다.")
    # 메세지 분리 에러
    FAILURE_SPLIT_MESSAGE = (-601, "메세지 분리에 실패하였습니다.")
    # 관리자 에러
    INVALID_ADMIN_ID = (-701, "관리자 인증에 실패했습니다.")


class ControlledException(RuntimeError):
    """
    예측 가능한 에러를 관리하기 위한 Exception Class
    """
    def __init__(self, error_code: ErrorCode):
        self.error_code = error_code