import logging

import ollama

from app.exception.incorrect_answer import IncorrectAnswer
from app.models.main_llm_model import main_llm
from app.models.split_llm_model import parsing_llm
from app.services.persona_store import persona_store
from app.models.database_client import database_client
from app.services.session_config import SessionConfig
from app.models.persona_llm_model import persona_llm_model
from app.models.postgres_client import postgres_client

from app.services.graph_rag_service import get_rag_prompt

try:
    ollama.pull("gemma3:4b")
except Exception:
    pass

def chat_full(prompt: str, model: str = "gemma3:4b") -> str:
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp["message"]["content"]

# NOTE: GraphRAG X
# def chat_stream(prompt: str, config: SessionConfig, model: str = "gemma3:4b"):
#     stream = ollama.generate(
#         model=model,
#         prompt=prompt,
#         stream=True
#     )
#     for chunk in stream:
#         yield chunk["response"]

# NOTE: GraphRAG O, Persona O
def chat_stream(prompt: str, config: SessionConfig):
    # TODO: ego_name으로 조회하는데, ego_id로 조회하고 있다. 수정할 것 (로직 자체는 이상 없음)
    ego_id:str = config.ego_id
    user_id:str = config.user_id
    session_id:str = f"{ego_id}@{user_id}"

    rag_prompt = get_rag_prompt(ego_id=ego_id, user_speak=prompt)
    persona = persona_store.get_persona(persona_id=ego_id)

    # NOTE. 비동기로 이전 에고 질문과 현재 사용자의 답변으로 문장을 추출한다.
    save_graphdb(session_id=session_id)

    for chunk in main_llm.get_chain().stream(
        input = {
            "input": prompt, # LLM에게 하는 질문을 프롬프트로 전달한다.
            "persona": persona,
            "related_story":rag_prompt, # 이전에 한 대화내역 중 관련 대화 내역을 프롬프트로 전달한다.
        },
        config={"configurable": {"session_id":f"{session_id}"}}
    ): 
        yield chunk.content

def save_persona(ego_id:str, session_history:str):
    """
    서버 메모리 내 대화내역을 바탕으로, ego의 persona를 강화한다.
    """
    # NOTE 1. 대화내역을 바탕으로 변경될 페르소나 정보를 추출한다.
    delta_persona = persona_llm_model.invoke(
        current_persona=persona_store.get_persona(persona_id=ego_id),
        session_history=session_history
    )

    # NOTE 2. 변경 값을 기존 페르소나에 적용한다.
    persona_store.update(persona_id=ego_id, delta_persona=delta_persona)  # 변경사항 업데이트

    # NOTE 3. 업데이트된 json을 postgres에 저장한다.
    postgres_client.update_persona(
        persona_id=ego_id,
        persona_json=persona_store.get_persona(persona_id=ego_id)
    )  # 데이터베이스에 업데이트 된 페르소나 저장

async def save_graphdb(session_id:str):
    """
    사용자의 답변이 들어올 시, 비동기로 사용자의 말을 GraphDatabase에 저장한다.
    """

    ego_id, user_id = session_id.split("@")

    # NOTE 1. 세션의 가장 마지막 대화 기록을 가져온다.
    # 이전 에고의 말을 가져오는 이유는, 사용자가 에고의 답변에 관한 내용을 말했을 수 있기 때문이다.
    # ex) 에고의 질문 or 사용자의 대명사 활용
    memory = main_llm.get_session_history(session_id=session_id)
    message = memory.messages[-1]
    human_message = f"{'human' if message.type == 'human' else 'ai' }: {message.content}"

    message = memory.messages.index[-2]
    ai_message = f"{'ai' if message.type == 'ai' else 'human' }: {message.content}"

    messages = [human_message, ai_message]

    # NOTE 2. 문장을 분리한다.
    try:
        splited_messages = parsing_llm.split_invoke(session_history=messages)
        logging.info(f"문장 분리 테스트: {splited_messages}")
    except IncorrectAnswer:
        return # 문장 분리 실패 시, 데이터는 저장하지 않는다.

    # NOTE 3. 에고에 맞게 삼중항을 저장한다.
    database_client.insert_messages_into_milvus(splited_messages=splited_messages, ego_id=ego_id)
