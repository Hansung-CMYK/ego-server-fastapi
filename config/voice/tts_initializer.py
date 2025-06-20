from config.voice.tts_infer import ensure_init


def load_tts_model(model_id: str, gpt_path: str, sovits_path: str) -> None:
    ensure_init(model_id, gpt_path, sovits_path)
