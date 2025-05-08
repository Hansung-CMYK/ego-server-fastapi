from app.models.singleton import main_llm


def get_chat_history_prompt(session_id:str)->str:
    """
        기존 서비스에선 다음과 같이 사용함
        ```
            chain.stream(
                {"input":user_speak,},
                config={"configurable": {"session_id":tenant_name}},
            })
        ```
    """
    return main_llm.get_session_history(sesssion_id=session_id)