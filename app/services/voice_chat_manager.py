from .voice_chat_session import VoiceChatSession

class VoiceChatSessionManager:
    def __init__(self):
        self.sessions: dict[str, VoiceChatSession] = {}

    def create_session(self, ws):
        session = VoiceChatSession(ws)
        self.sessions[session.id] = session
        return session

    def remove_session(self, sid: str):
        if sid in self.sessions:
            self.sessions[sid].stop()
            del self.sessions[sid]
