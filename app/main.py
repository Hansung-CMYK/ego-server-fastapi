import importlib.util
import sys
import os
from fastapi import FastAPI

app = FastAPI()

api_file_path = "./modules/GPT-SoVITS/api.py"
gpt_sovits_root = os.path.abspath(os.path.dirname(api_file_path))

if gpt_sovits_root not in sys.path:
    sys.path.insert(0, gpt_sovits_root)

spec = importlib.util.spec_from_file_location("gpt_sovits_api", api_file_path)
gpt_sovits_api = importlib.util.module_from_spec(spec)
sys.modules["gpt_sovits_api"] = gpt_sovits_api
spec.loader.exec_module(gpt_sovits_api)

if hasattr(gpt_sovits_api, "app"):
    app.include_router(gpt_sovits_api.app)
else:
    raise ImportError("modules/GPT-SoVITS/api.py에는 'app' 객체가 없습니다.")

