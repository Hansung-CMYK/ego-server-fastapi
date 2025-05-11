import ollama

from app.services.persona_store import PersonaStore
from app.services.session_config import SessionConfig

from app.models.singleton import main_llm
from app.services.graph_rag_service import get_rag_prompt
from app.models.postgres_client import postgres_client

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
def chat_stream(prompt: str, config: SessionConfig, model: str = "gemma3:4b"):

    rag_prompt = get_rag_prompt(ego_name=config.ego_id, user_speak=prompt)
    # TODO: dict에 존재하는지 찾아보고 없으면 조회하기 로직으로 수정할 것
    search_result = postgres_client.select_persona_to_id(persona_id=config.ego_id)
    persona_store = PersonaStore(search_result[0], search_result[1], search_result[2])

    for chunk in main_llm.get_chain().stream(
        model = model,
        input = {
            "input": prompt, # LLM에게 하는 질문을 프롬프트로 전달한다.
            "persona": persona_store.get_persona(),
            "related_story":rag_prompt, # 이전에 한 대화내역 중 관련 대화 내역을 프롬프트로 전달한다.
        },
        config={"configurable": {"session_id":f"{config.ego_id}@{config.user_id}"}}
    ): 
        yield chunk.content