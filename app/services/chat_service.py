import ollama

from app.exception.incorrect_answer import IncorrectAnswer
from app.models.main_llm_model import main_llm
from app.models.split_llm_model import parsing_llm
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
    # TODO: ego_name으로 조회하는데, ego_id로 조회하고 있다. 수정할 것 (로직 자체는 이상 없음)
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
    서버 메모리 내 대화내역을 바탕으로, ego의 persona를 강화한다.
    """
    # 하루에 한번 하는 로직이므로 함수 내부에 import
    from app.models.persona_llm_model import persona_llm_model
    from app.models.postgres_client import postgres_client

    # NOTE 1. userId, egoId, 대화내역을 불러온다.
    store_keys = main_llm.get_store_keys() # 모든 채팅방의 대화 내역을 저장한다.

    # 모든 채팅방 대화 내역을 기준으로 반복한다.
    # TODO: 유저 단위가 아닌 채팅방 단위이므로, 현재 매우 비효율적인 로직이다. (user_id를 기반으로 묶어 저장하는 것이 정석)
    for session_id in store_keys:
        ego_id, user_id = session_id.split("@")

        # 세션 정보로 해당 채팅방의 대화 내역을 불러온다.
        session_history = main_llm.get_human_messages_in_memory(session_id=session_id)

        # NOTE 2. 대화내역을 바탕으로 변경될 페르소나를 추출한다.
        delta_persona = persona_llm_model.invoke(
            current_persona=persona_store.get_persona(persona_id=ego_id),
            session_history=session_history
        )

        # NOTE 3. 변경 값을 기존 페르소나에 적용한다.
        persona_store.update(persona_id=ego_id, delta_persona=delta_persona)  # 변경사항 업데이트

        # NOTE 4. 업데이트된 json을 postgres에 저장한다.
        postgres_client.update_persona(
            persona_name=ego_id,
            persona_json=persona_store.get_persona(persona_id=ego_id)
        )  # 데이터베이스에 업데이트 된 페르소나 저장
    return

# def save_graphdb():
#     # NOTE 1. userId, egoId, 대화내역을 불러온다.
#     message_history = main_llm.get_human_messages_in_memory(session_id=tenant_name)
#
#     # NOTE 2. 대화내역을 단일 사실을 가진 문장으로 분리한다.
#     try:
#         splited_messages = parsing_llm.split_invoke(message_history=message_history)
#         print("\n")
#         print("========================================")
#         print(f"splited_messages: {splited_messages}")
#     except IncorrectAnswer:
#         return
#         # continue # TODO: 대화 내역을 불러와서 loop로 만들 것이기 때문에, 추후 continue를 사용하게 된다.
#
#     # NOTE 3. 단일 문장을 삼중항으로 분리하여 각 에고에 알맞게 저장한다.
#     from app.models.database_client import database_client
#     database_client.insert_messages_into_milvus(splited_messages=splited_messages, partition_name=tenant_name)
#     return