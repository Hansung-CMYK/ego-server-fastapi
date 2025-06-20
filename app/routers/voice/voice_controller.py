import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config.session.session_config import SessionConfig
from config.voice.voice_chat_manager import VoiceChatSessionManager

router = APIRouter()
session_manager = VoiceChatSessionManager()

import logging

logger = logging.getLogger(__name__)


@router.websocket("/ws/voice-chat")
async def websocket_voice_chat(ws: WebSocket):
    await ws.accept()
    user_id = ws.query_params.get("user_id", "anonymous")
    ego_id = ws.query_params.get("ego_id", "anonymous")
    spk = ws.query_params.get("spk", "anonymous")
    chat_room_id = ws.query_params.get("chat_room_id", "anonymous")

    config = SessionConfig(user_id, ego_id)
    config.spk = spk
    config.chat_room_id = chat_room_id
    # NOTE uid로 refer_path 조회 및 적용

    ego_user_id = _get_user_id(ego_id)

    if user_id is None:
        logger.error('user_id 조회 실패')
    else:
        refer_path = _get_refer_path(ego_user_id)
        if refer_path is not None and 'karina' not in refer_path:
            config.refer_path = refer_path
            config.prompt_text = "2025년 캡스톤디자인 에고 시연입니다."

    if config is None:
        raise Exception("spk was None")
    if chat_room_id is None:
        raise Exception("chat_room_id was None")

    session = session_manager.create_session(ws, config)

    try:
        while True:
            msg = await ws.receive_bytes()
            session.handle_audio(msg)
    except WebSocketDisconnect:
        pass
    finally:
        session_manager.remove_session(session.id)


def _get_user_id(ego_id: str):
    url = f"http://localhost:30080/api/v1/ego/{ego_id}/owner"
    try:
        response = httpx.get(url, timeout=5.0)
    except httpx.RequestError as e:
        return {"error": f"Request failed: {str(e)}"}

    if response.status_code != 200:
        return {"error": f"Request failed with status {response.status_code}"}

    json_data = response.json()

    # 응답에서 uid 추출
    if json_data.get("code") == 200 and "data" in json_data:
        uid = json_data["data"].get("uid")
        return uid
    else:
        return None


def _get_refer_path(user_id: str):
    url = f"http://localhost:30080/api/v1/ego/voice/{user_id}"
    try:
        response = httpx.get(url, timeout=3.0)
    except httpx.RequestError as e:
        return {"error": f"Request failed: {str(e)}"}

    if response.status_code != 200:
        return {"error": f"Request failed with status {response.status_code}"}

    json_data = response.json()

    if json_data.get("code") == 200 and "data" in json_data:
        voice_url = json_data["data"].get("voiceUrl")
        return voice_url
    else:
        return None

