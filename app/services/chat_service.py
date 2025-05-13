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

    rag_prompt = get_rag_prompt(ego_id=ego_id, user_speak=prompt)
    persona = persona_store.get_persona(persona_id=ego_id)

    for chunk in main_llm.get_chain().stream(
        input = {
            "input": prompt, # LLM에게 하는 질문을 프롬프트로 전달한다.
            "persona": persona,
            "related_story":rag_prompt, # 이전에 한 대화내역 중 관련 대화 내역을 프롬프트로 전달한다.
        },
        config={"configurable": {"session_id":f"{ego_id}@{user_id}"}}
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

def save_graphdb(ego_id:str, session_history:str):
    """
    서버 메모리 내 대화내역을 바탕으로, ego의 맥락 정보를 추출한다.
    """
    # NOTE 1. 대화내역을 단일 사실을 가진 문장으로 분리한다.
    try:
        splited_messages = parsing_llm.split_invoke(session_history=session_history)
    except IncorrectAnswer:
        return # 문장 분리 실패 시, 데이터는 저장하지 않는다.

    # NOTE 2. 단일 문장을 삼중항으로 분리하여 각 에고에 알맞게 저장한다.
    database_client.insert_messages_into_milvus(splited_messages=splited_messages, ego_id=ego_id)
