import torch
import importlib
from app.services.tts_model_registry import register_model, has_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_half = True 

async def ensure_init(
    model_id: str,
    gpt_path: str,
    sovits_path: str
) -> None:
    if has_model(model_id):
        return

    gsv = importlib.import_module("gpt_sovits_api")

    get_gpt_weights = getattr(gsv, "get_gpt_weights")
    gpt = get_gpt_weights(gpt_path)
    gpt.t2s_model.to(device)
    if is_half:
        gpt.t2s_model.half()

    get_sovits_weights = getattr(gsv, "get_sovits_weights")
    sovits = get_sovits_weights(sovits_path)
    sovits.vq_model.to(device)
    if is_half:
        sovits.vq_model.half()

    register_model(model_id, gpt, sovits)

    Speaker = getattr(gsv, "Speaker")
    speaker = Speaker(name=model_id, gpt=gpt, sovits=sovits)
    gsv.speaker_list[model_id] = speaker
