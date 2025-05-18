import ollama

from app.models.main_llm import main_llm
from app.models.split_llm import split_llm
from app.models.milvus_database import milvus_database
from app.services.graph_rag_service import get_rag_prompt
from app.services.persona_store import persona_store
from app.services.session_config import SessionConfig

try:
    ollama.pull("gemma3:4b")
except Exception:
    pass

# NOTE: GraphRAG O, Persona O
def chat_stream(prompt: str, config: SessionConfig):
    # TODO: ego_name으로 조회하는데, ego_id로 조회하고 있다. 수정할 것 (로직 자체는 이상 없음)
    ego_id:str = config.ego_id
    user_id:str = config.user_id
    session_id:str = f"{ego_id}@{user_id}"

    # NOTE. 비동기로 이전 에고 질문과 현재 사용자의 답변으로 문장을 추출한다.
    # asyncio.create_task(save_graphdb(session_id=session_id, user_answer=prompt))

    rag_prompt = get_rag_prompt(ego_id=ego_id, user_speak=prompt)
    persona = persona_store.get_persona(ego_id=ego_id)

    for chunk in main_llm.get_chain().stream(
        input = {
            "input": prompt, # LLM에게 하는 질문을 프롬프트로 전달한다.
            "persona": persona,
            "related_story":rag_prompt, # 이전에 한 대화내역 중 관련 대화 내역을 프롬프트로 전달한다.
        },
        config={"configurable": {"session_id":f"{session_id}"}}
    ):
        yield chunk.content

async def save_graphdb(session_id:str, user_answer:str):
    """
    사용자의 답변이 들어올 시, 비동기로 사용자의 말을 GraphDatabase에 저장한다.
    """

    ego_id, user_id = session_id.split("@")

    # NOTE 1. 세션의 가장 마지막 대화 기록을 가져온다.
    # 이전 에고의 말을 가져오는 이유는, 사용자가 에고의 답변에 관한 내용을 말했을 수 있기 때문이다.
    # ex) 에고의 질문 or 사용자의 대명사 활용
    memory = main_llm.get_session_history(session_id=session_id)

    message = memory.messages[-1] # 가장 마지막 말인 에고 질문 추출
    ai_message = f"{'human' if message.type == 'human' else 'ai'}: {message.content}"
    human_message = f"human: {user_answer}" # 사용자가 말한 답변 저장

    messages = [ai_message, human_message]

    # NOTE 2. 문장을 분리한다.
    splited_messages = split_llm.invoke(session_history=messages)
    if len(splited_messages) == 0: return # 문장 분리 실패 시, 데이터는 저장하지 않는다.

    # NOTE 3. 에고에 맞게 삼중항을 저장한다.
    # TODO: 유저 정보로 해당 유저의 에고 아이디 조회가 필요하다. (하단 코드는 잘못된 로직)
    # TODO: `ego_id_of_user = <api>(user_id)` BE API에 필요
    milvus_database.insert_messages_into_milvus(splited_messages=splited_messages, ego_id=ego_id)