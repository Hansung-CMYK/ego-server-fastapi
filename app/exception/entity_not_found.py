class EntityNotFound(Exception):
    """
        Milvus에서 특정 id로 entity를 조회하지 못했을 때 발생하는 에러이다.
    """
    pass