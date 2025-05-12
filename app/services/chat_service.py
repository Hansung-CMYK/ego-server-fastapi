import ollama

from app.models.main_llm_model import main_llm
from app.services.persona_store import persona_store
from app.services.session_config import SessionConfig

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

    rag_prompt = get_rag_prompt(ego_name=config.ego_id, user_speak=prompt)
    persona = persona_store.get_persona(persona_id=config.ego_id)

    for chunk in main_llm.get_chain().stream(
        input = {
            "input": prompt, # LLM에게 하는 질문을 프롬프트로 전달한다.
            "persona": persona,
            "related_story":rag_prompt, # 이전에 한 대화내역 중 관련 대화 내역을 프롬프트로 전달한다.
        },
        config={"configurable": {"session_id":f"{config.ego_id}@{config.user_id}"}}
    ): 
        yield chunk.content

def save_persona():
    """
    서버 메모리 내 대화내역을 취합하여, 에고를 생성한다.
    """
    session_history = main_llm.get_session_history_to_str(tenant_name)

    from app.models.persona_llm_model import persona_llm_model
    delta_persona = persona_llm_model.invoke(
        current_persona=persona_store.get_persona(),
        session_history=session_history
    )
    print("=== Delta Persona ===")
    print(delta_persona)
    persona_store.update(delta_persona)  # 변경사항 업데이트

    from app.models.postgres_client import postgres_client
    postgres_client.update_persona(
        persona_name=persona_store.persona_name,
        persona_json=persona_store.get_persona()
    )  # 데이터베이스에 업데이트 된 페르소나 저장

    return

def save_graphdb():
    return