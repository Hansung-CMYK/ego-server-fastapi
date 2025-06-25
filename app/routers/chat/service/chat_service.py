import asyncio
import threading

from app.internal.logger.logger import logger
from app.routers.chat.service import milvus_service, graph_rag_service
from app.routers.chat.service.persona_store import persona_store
from config.external import hub_api
from config.llm.main_llm import main_llm
from config.llm.split_llm import SplitLLM
from config.session.session_config import SessionConfig

MAIN_LOOP = asyncio.new_event_loop()
threading.Thread(target=MAIN_LOOP.run_forever, daemon=True).start()

def worker(session_id:str, user_message:str):
    """
    요약:
        save_graphdb를 수행하는 코루틴을 생성하는 함수

    Parameters:
        session_id(str): 저장하고 싶은 대화 내역이 들어있는 세션 아이디
        user_message(str): 사용자의 메세지
    """
    asyncio.run_coroutine_threadsafe(save_graphdb(session_id=session_id, user_message=user_message), MAIN_LOOP)

def chat_stream(user_message: str, config: SessionConfig):
    """
    요약:
        AI가 지식 그래프와 페르소나 정보를 활용해 답변할 수 있게하는 함수이다.

    Parameters:
        user_message(str): 사용자 메세지
        config(SessionConfig): 세션(웹 소켓) 정보
    """
    ego_id:str = config.ego_id
    user_id:str = config.user_id
    session_id:str = f"{ego_id}@{user_id}"

    # NOTE 0. 비동기로 사용자의 답변을 지식 그래프에 추가한다.
    worker(session_id=session_id, user_message=user_message)

    # NOTE 1. 에고의 페르소나를 적용한다.
    persona = persona_store.get_persona(ego_id=ego_id)

    # NOTE 2. 에고가 가진 지식 그래프에서 정보를 조회한다.
    rag_prompt = graph_rag_service.get_rag_prompt(ego_id=ego_id, user_message=user_message)

    # NOTE 3. 사용자의 이름 추출
    # my_ego = get_ego(user_id=user_id)

    # NOTE 4. 에고의 답변을 청크 단위로 출력한다.
    for chunk in main_llm.main_stream(
        user_message= user_message,
        persona = persona,
        rag_prompt = rag_prompt,
        session_id = session_id
    ):
        yield chunk

async def save_graphdb(session_id:str, user_message:str):
    """
    요약:
        사용자의 답변이 들어올 시, 비동기로 사용자의 말을 GraphDatabase에 저장한다.

    Parameters:
        session_id(str): 대화 내역이 저장된 세션의 아이디
        user_message(str): 사용자 메세지
    """
    # 세션의 정보를 분리한다.
    ego_id, user_id = session_id.split("@")

    # NOTE 1. 세션의 가장 마지막 대화 기록을 가져온다.
    # 이전 에고의 말을 가져오는 이유는, 사용자가 에고의 답변에 관한 내용을 말했을 수 있기 때문이다.
    # ex) 에고의 질문 or 사용자의 대명사 활용
    memory = main_llm.get_session_history(session_id=session_id)

    input:str = f"HUMAN: {user_message}"
    for message in reversed(memory.messages):
        if message.type == "ai":
            input.join(f"\nAI: {message.content}")
            break
        input.join(f"\nHUMAN: {message.content}")

    # NOTE 2. 문장을 분리한다.
    split_messages = SplitLLM().invoke({
        "input": input
    })

    # LOG. 시연용 로그
    logger.info(f"\n\nPOST: api/v1/chat [저장할 단일 문장들]\n{input}\n")

    # NOTE 3. 에고에 맞게 삼중항을 저장한다.
    my_ego = hub_api.get_ego(user_id=user_id)

    milvus_service.insert_messages(single_sentences=split_messages, passage=input, ego_id=str(my_ego["id"]))
    logger.info("save_graphdb success!")