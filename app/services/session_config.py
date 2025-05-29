class SessionConfig:
    def __init__(
        self,
        user_id : str,
        ego_id : str,
    ):
        self.user_id = user_id
        self.ego_id = ego_id
        self.session_id = f"{ego_id}@{user_id}"
        self.spk: str = None
        self.chat_room_id : int = None
        self.refer_path : str = "/home/keem/refer/karina.wav"
        self.prompt_text = "내 마음에 드는 거 있으면 낭독해줄게?"