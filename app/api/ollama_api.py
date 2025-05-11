import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse

from app.services.graph_rag_service import get_rag_prompt
from app.services.ollama_service import chat_full, chat_stream

router = APIRouter()

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@router.post(
    "/ollama",
    response_model=ChatResponse,
    summary="Ollama 모델에 메시지 전달하고 전체 응답 받기",
    tags=["ollama"],
)
async def http_ollama(body: ChatRequest) -> JSONResponse:
    """
    요청 예시:
    {
      "message": "너의 이름은 무엇이니"
    }
    """
    resp = chat_full(body.message)
    return JSONResponse({"response": resp})

@router.websocket("/ws/ollama")
async def ws_ollama(ws: WebSocket):
    await ws.accept()
    try:
        raw = await ws.receive_text()
        payload = json.loads(raw)
        prompt = payload.get("message")
        if not prompt:
            await ws.send_text(json.dumps({"error": "message 필요"}))
            return
        for chunk in chat_stream(prompt):
            await ws.send_text(json.dumps({"chunk": chunk}))
        await ws.send_text(json.dumps({"done": True}))
    except WebSocketDisconnect:
        return

""" ########## 김명준이 추가한 함수 ########## """
from app.models.singleton import main_llm
from langchain_core.messages import SystemMessage

class PromptRequest(BaseModel):
    user_speak: str

@router.post("/ws/ollama_temp/{ego_name}/{session_id}")
async def ws_ollama_temp(ego_name: str, session_id: str, body: PromptRequest):
    """
    예시
    ego_name = "ego"
    session_id = "1234"
    """
    rag_prompt = get_rag_prompt(ego_name=ego_name, user_speak=body.user_speak)

    return main_llm.get_chain().invoke({
                "input":body.user_speak, # LLM에게 하는 질문을 프롬프트로 전달한다.
                "related_story":[SystemMessage(content=rag_prompt)], # 이전에 한 대화내역 중 관련 대화 내역을 프롬프트로 전달한다.
            },
            config={"configurable": {"session_id":f"{ego_name}@{session_id}"}}, # 일일 대화 내역 저장
        )["content"]