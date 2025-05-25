from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.voice.voice_chat_manager import VoiceChatSessionManager
from app.services.session_config import SessionConfig
router = APIRouter()
session_manager = VoiceChatSessionManager()

@router.websocket("/ws/voice-chat")
async def websocket_voice_chat(ws: WebSocket):
    await ws.accept()
    user_id = ws.query_params.get("user_id", "anonymous")
    ego_id = ws.query_params.get("ego_id", "anonymous")
    spk = ws.query_params.get("spk", "anonymous")
    chat_room_id = int(ws.query_params.get("chat_root_id", "anonymous"))

    config = SessionConfig(user_id, ego_id)
    config.spk = spk
    config.chat_room_id = chat_room_id

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

