import torch
import importlib

from deprecated import deprecated

import GPUtil

from config.voice.tts_model_registry import register_model, has_model


def pick_least_used_gpu() -> int | None:
    if not torch.cuda.is_available():
        return None
    gpus = GPUtil.getGPUs()
    count = torch.cuda.device_count()
    valid = [g.id for g in gpus if g.id < count]
    if not valid:
        return None
    return min(valid, key=lambda i: next(g.memoryUtil for g in gpus if g.id == i))

_best_gpu = pick_least_used_gpu()
if _best_gpu is not None:
    device = torch.device(f"cuda:{_best_gpu}")
else:
    device = torch.device("cpu")


is_half = True

@deprecated
async def ensure_init(
    model_id: str,
    gpt_path: str,
    sovits_path: str,
    refer_path: str,
    refer_text: str,
    refer_language: str
) -> None:
    
    if refer_path == None or refer_text == None or refer_language == None:
        raise Exception

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
    DefaultRefer = getattr(gsv, "DefaultRefer")
    speaker = Speaker(name=model_id, gpt=gpt, sovits=sovits, default_refer = DefaultRefer(refer_path, refer_text, refer_language))
    gsv.speaker_list[model_id] = speaker


async def ensure_init_v2(
    model_id: str,
    gpt_path: str,
    sovits_path: str,
    refer_path: str,
    refer_text: str,
    refer_language: str
) -> None:
    if not (refer_path and refer_text and refer_language):
        raise ValueError("refer_path, refer_text, refer_language 모두 필요합니다.")

    if has_model(model_id):
        return

    gsv = importlib.import_module("gpt_sovits_api")

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    custom_config = {
        "version": "v4",
        "device": str(device),
        "is_half": is_half,
        "t2s_weights_path": gpt_path,
        "vits_weights_path": sovits_path,
    }
    tts_conf = TTS_Config({"custom": custom_config})
    tts_pipe = TTS(tts_conf)

    register_model(model_id, tts_pipe)

    Speaker      = getattr(gsv, "Speaker")
    DefaultRefer = getattr(gsv, "DefaultRefer")

    speaker = Speaker(
        name=model_id,
        tts_pipeline=tts_pipe,
        default_refer=DefaultRefer(refer_path, refer_text, refer_language)
    )
    gsv.speaker_list[model_id] = speaker
