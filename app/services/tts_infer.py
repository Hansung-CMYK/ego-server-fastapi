from pathlib import Path
import sys
import importlib.util
from app.services.tts_model_registry import register_model, has_model

BASE_DIR = Path(__file__).resolve().parents[2]
GSV_DIR = BASE_DIR / "modules" / "GPT-SoVITS"
GSV_API_PATH = GSV_DIR / "api.py"

# GPT_SoVITS 내 모듈 import 가능하도록 sys.path 추가
sys.path.insert(0, str(GSV_DIR))                 # api.py 기준 상대 import 대응
sys.path.insert(0, str(GSV_DIR / "GPT_SoVITS"))  # text, f5_tts 등의 절대 import 대응

# 동적 import
spec = importlib.util.spec_from_file_location("gsv_api", GSV_API_PATH)
gsv_api = importlib.util.module_from_spec(spec)
sys.modules["gsv_api"] = gsv_api
spec.loader.exec_module(gsv_api)

get_tts_wav = gsv_api.get_tts_wav

def ensure_init(model_id: str, gpt_path: str, sovits_path: str) -> None:
    if has_model(model_id):
        return
    if model_id == "default" and "default" in gsv_api.speaker_list:
        speaker = gsv_api.speaker_list["default"]
        register_model(model_id, speaker.gpt, speaker.sovits)
        return
    gsv_api.change_gpt_sovits_weights(gpt_path=gpt_path, sovits_path=sovits_path)
    speaker = gsv_api.speaker_list["default"]
    register_model(model_id, speaker.gpt, speaker.sovits)

