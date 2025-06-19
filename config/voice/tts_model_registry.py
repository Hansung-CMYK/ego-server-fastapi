models: dict[str, dict] = {}

def has_model(model_id: str) -> bool:
    return model_id in models

def register_model(model_id: str, tts_pipeline) -> None:
    models[model_id] = { "tts": tts_pipeline }

def get_tts_pipeline(model_id: str):
    entry = models.get(model_id)
    if not entry:
        raise KeyError(f"모델 '{model_id}' 이(가) 등록되지 않았습니다.")
    return entry["tts"]
