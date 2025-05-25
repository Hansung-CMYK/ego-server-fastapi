import os
import sys
import importlib.util

from fastapi import FastAPI, APIRouter
from app.api import chat_api, voice_chat_api, fal_api, diary_api, persona_api
from app.exception.exception_handler import register_exception_handlers
from app.services.voice.tts_infer import ensure_init, ensure_init_v2

from contextlib import asynccontextmanager

import app.services.kafka.kafka_handler as kh
import asyncio

import logging
logging.basicConfig(level=logging.INFO)

here = os.path.dirname(__file__)
api_file_path = os.path.abspath(os.path.join(here, "../modules/GPT-SoVITS/api_v2.py"))
gpt_sovits_root = os.path.dirname(api_file_path)
gpt_sovits_sub  = os.path.join(gpt_sovits_root, "GPT_SoVITS")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GPT_SOVITS_ROOT = os.path.join(BASE_DIR, "modules", "GPT-SoVITS", "GPT_SoVITS", "pretrained_models")
WEIGHT_PATH = os.path.join(BASE_DIR, "modules", "GPT-SoVITS", "weights")

for path in (gpt_sovits_root, gpt_sovits_sub):
    if path not in sys.path:
        sys.path.insert(0, path)

spec = importlib.util.spec_from_file_location("gpt_sovits_api", api_file_path)
gpt_sovits_api = importlib.util.module_from_spec(spec)
sys.modules["gpt_sovits_api"] = gpt_sovits_api
spec.loader.exec_module(gpt_sovits_api)

user_home = os.path.expanduser("~")
REFER_DIRECTORY = os.path.join(user_home, "refer")

async def init_models():
    await ensure_init_v2(
        "karina",
        # ckpt
        os.path.join(WEIGHT_PATH, "karina-v4-2505242-e20.ckpt"),
        # pth
        os.path.join(WEIGHT_PATH, "karina-v4-2505242_e8_s200_l64.pth"),
        os.path.join(REFER_DIRECTORY, "karina.wav"),
        "내 마음에 드는 거 있으면 낭독해줄게?",
        "ko"
    )

    await ensure_init_v2(
        "ralo",
        # ckpt
        os.path.join(WEIGHT_PATH, "ralo-v4-2505251-e20.ckpt"),
        # pth
        os.path.join(WEIGHT_PATH, "ralo-v4-2505251_e8_s344_l64.pth"),
        os.path.join(REFER_DIRECTORY, "ralo.wav"),
        "안 나오네? 이거 누가 좀 찾아주세요 내가 이상한 사람 되잖아 저기 누가 좀 찾아주십시오 제... 제가",
        "ko"
    )

async def on_startup():
    await kh.init_kafka()
    asyncio.create_task(kh.consume_loop())
    return 

async def on_shutdown():
    await kh.shutdown_kafka()
    return 

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_models()
    await on_startup()
    
    yield
    await on_shutdown()
    

app = FastAPI(lifespan=lifespan)

tts_router = APIRouter(prefix="/tts", tags=["tts"])

for route in gpt_sovits_api.APP.routes:
    if route.path == "/":
        continue
    tts_router.routes.append(route)

register_exception_handlers(app)

app.include_router(tts_router)

app.include_router(chat_api.router, prefix="/api", tags=["admin"])
app.include_router(chat_api.router, prefix="/api", tags=["chat"])
app.include_router(voice_chat_api.router, prefix="/api", tags=["voice-chat"])
app.include_router(fal_api.router, prefix="/api", tags=["image"])
app.include_router(diary_api.router, prefix="/api", tags=["diary"])
app.include_router(persona_api.router, prefix="/api", tags=["persona"])
