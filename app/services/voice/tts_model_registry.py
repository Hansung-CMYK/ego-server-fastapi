from typing import Dict, Tuple

_TTS_REGISTRY: Dict[str, Tuple[object, object]] = {}

def register_model(model_id: str, gpt_obj: object, sovits_obj: object) -> None:
    _TTS_REGISTRY[model_id] = (gpt_obj, sovits_obj)

def has_model(model_id: str) -> bool:
    return model_id in _TTS_REGISTRY

def get_model(model_id: str) -> Tuple[object, object]:
    return _TTS_REGISTRY[model_id]

