import torch
from app.services.tts_model_registry import register_model, has_model
import importlib

from fastapi import Query
from fastapi.responses import StreamingResponse, JSONResponse

from deprecated import deprecated

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_half = True

def ensure_init(model_id: str, gpt_path: str, sovits_path: str) -> None:
    gsv = importlib.import_module("gpt_sovits_api")

    if has_model(model_id):
        return

    if model_id == "default" and "default" in gsv.speaker_list:
        speaker = gsv.speaker_list["default"]
        register_model(model_id, speaker.gpt, speaker.sovits)
        return

    gsv.change_gpt_sovits_weights(gpt_path=gpt_path, sovits_path=sovits_path)
    speaker = gsv.speaker_list["default"]

    speaker.gpt.t2s_model.to(device)
    if is_half:
        speaker.gpt.t2s_model.half()

    speaker.sovits.vq_model.to(device)
    if is_half:
        speaker.sovits.vq_model.half()

    register_model(model_id, speaker.gpt, speaker.sovits)
