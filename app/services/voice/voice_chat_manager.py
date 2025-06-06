from .voice_chat_handler import VoiceChatHandler
from app.services.session_config import SessionConfig

class VoiceChatSessionManager:
    def __init__(self):
        self.sessions: dict[str, VoiceChatHandler] = {}

    def create_session(self, ws, config : SessionConfig):
        session = VoiceChatHandler(ws, config)
        self.sessions[session.id] = session
        return session

    def remove_session(self, sid: str):
        if sid in self.sessions:
            self.sessions[sid].stop()
            del self.sessions[sid]
