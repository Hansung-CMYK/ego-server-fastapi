import os
import sys
import importlib.util
from fastapi import FastAPI

from app.api import ollama_api, voice_chat_api
from app.services.tts_infer import ensure_init

app = FastAPI()

here = os.path.dirname(__file__)
api_file_path = os.path.abspath(os.path.join(here, "../modules/GPT-SoVITS/api.py"))
gpt_sovits_root = os.path.dirname(api_file_path)
gpt_sovits_sub = os.path.join(gpt_sovits_root, "GPT_SoVITS")

for path in (gpt_sovits_root, gpt_sovits_sub):
    if path not in sys.path:
        sys.path.insert(0, path)

spec = importlib.util.spec_from_file_location("gpt_sovits_api", api_file_path)
gpt_sovits_api = importlib.util.module_from_spec(spec)
sys.modules["gpt_sovits_api"] = gpt_sovits_api
spec.loader.exec_module(gpt_sovits_api)

@app.on_event("startup")
def init_models():
    model_id = "default"
    gpt_path = "/path/to/gpt_weights"
    sovits_path = "/path/to/sovits_weights"
    ensure_init(model_id, gpt_path, sovits_path)

if hasattr(gpt_sovits_api, "app"):
    for route in gpt_sovits_api.app.routes:
        app.router.routes.append(route)
else:
    raise ImportError("modules/GPT-SoVITS/api.py에는 'app' 객체가 없습니다.")

app.include_router(ollama_api.router, prefix="/api")
app.include_router(voice_chat_api.router, prefix="/api")

