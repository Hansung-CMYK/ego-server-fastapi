import threading

from app.models.chat.main_llm import main_llm
from app.models.database.milvus_database import milvus_database
from app.services.chatting.graph_rag_service import get_rag_prompt
from app.services.chatting.persona_store import persona_store
from app.services.diary.diary_service import SPRING_URI
from app.models.txtnorm.split_llm import split_llm
from app.services.session_config import SessionConfig
import asyncio
import requests

MAIN_LOOP = asyncio.new_event_loop()
threading.Thread(target=MAIN_LOOP.run_forever, daemon=True).start()

def worker(session_id:str, user_answer:str):
    asyncio.run_coroutine_threadsafe(save_graphdb(session_id=session_id, user_answer=user_answer), MAIN_LOOP)

# NOTE: GraphRAG O, Persona O
def chat_stream(prompt: str, config: SessionConfig):
    ego_id:str = config.ego_id
    user_id:str = config.user_id
    session_id:str = f"{ego_id}@{user_id}"

    # NOTE. 비동기로 이전 에고 질문과 현재 사용자의 답변으로 문장을 추출한다.
    worker(session_id=session_id, user_answer=prompt)

    rag_prompt = get_rag_prompt(ego_id=ego_id, user_speak=prompt)
    persona = persona_store.get_persona(ego_id=ego_id)

    for chunk in main_llm.stream(
        input = prompt,
        persona = persona,
        rag_prompt = rag_prompt,
        session_id = session_id
    ):
        yield chunk

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

    ai_message = f"AI: {message.content}"
    human_message = f"HUMAN: {user_answer}" # 사용자가 말한 답변 저장

    input = "\n".join([ai_message, human_message])

    # NOTE 2. 문장을 분리한다.
    splited_messages = split_llm.invoke(input=input)
    if len(splited_messages) == 0: return # 문장 분리 실패 시, 데이터는 저장하지 않는다.

    # NOTE 3. 에고에 맞게 삼중항을 저장한다.
    # user_id로 my_ego 추출
    url = f"{SPRING_URI}/api/v1/ego/user/{user_id}"
    response = requests.get(url)
    my_ego = response.json()["data"]

    milvus_database.insert_messages_into_milvus(splited_messages=splited_messages, ego_id=my_ego["id"])
    print(f"save_graphdb success: {splited_messages}")