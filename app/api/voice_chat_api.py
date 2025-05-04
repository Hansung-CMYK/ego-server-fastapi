from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.voice_chat_manager import VoiceChatSessionManager

router = APIRouter()
session_manager = VoiceChatSessionManager()

@router.websocket("/ws/voice-chat")
async def websocket_voice_chat(ws: WebSocket):
    await ws.accept()
    session = session_manager.create_session(ws)

    try:
        while True:
            msg = await ws.receive_bytes()
            session.handle_audio(msg) 
    except WebSocketDisconnect:
        pass
    finally:
        session_manager.remove_session(session.id)

