class PersonaNotFound(Exception):
    """
        사용자가 잘못된 응답을 제시했을 때, 발생하는 에러이다.
        주로 LLM이 삼중항으로 분리하지 못할 때, 발생한다.
    """
    pass