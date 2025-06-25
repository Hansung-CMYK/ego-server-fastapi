from config.common.common_session import CommonSession
from .voice_chat_handler import VoiceChatHandler


class VoiceChatSessionManager:
    def __init__(self):
        self.sessions: dict[str, VoiceChatHandler] = {}

    def create_session(self, ws, config : CommonSession):
        session = VoiceChatHandler(ws, config)
        self.sessions[session.id] = session
        return session

    def remove_session(self, sid: str):
        if sid in self.sessions:
            self.sessions[sid].stop()
            del self.sessions[sid]
