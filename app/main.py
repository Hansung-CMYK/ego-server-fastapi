import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

import config.kafka.kafka_handler as kh
from app.internal.admin import admin_controller
from app.internal.exception.exception_handler import \
    register_exception_handlers
from app.routers.chat import chat_controller
from app.routers.diary import diary_controller
from app.routers.fal import fal_controller
from app.routers.persona import persona_controller
from app.routers.stt import stt_controller
from app.routers.tone import tone_controller
from app.routers.tts import tts_controller
from app.routers.voice import voice_controller
from config.voice.tts_infer import ensure_init_v2

logging.basicConfig(level=logging.INFO)

here = os.path.dirname(__file__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GPT_SOVITS_ROOT = os.path.join(BASE_DIR, "modules", "GPT-SoVITS", "GPT_SoVITS", "pretrained_models")
WEIGHT_PATH = os.path.join(BASE_DIR, "modules", "GPT-SoVITS", "weights")

user_home = os.path.expanduser("~")
REFER_DIRECTORY = os.path.join(user_home, "refer")

async def init_models():
    await ensure_init_v2(
        "default",
        # ckpt
        os.path.join(WEIGHT_PATH, "karina-v4-2505242-e20.ckpt"),
        # pth
        os.path.join(WEIGHT_PATH, "karina-v4-2505242_e8_s200_l64.pth"),
        os.path.join(REFER_DIRECTORY, "karina.wav"),
        "내 마음에 드는 거 있으면 낭독해줄게?",
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


register_exception_handlers(app)

app.include_router(stt_controller.router, prefix='/api', tags=['stt'])
app.include_router(tts_controller.router, prefix="/api", tags=["tts"])
app.include_router(admin_controller.router, prefix="/api", tags=["admin"])
app.include_router(chat_controller.router, prefix="/api", tags=["chat"])
app.include_router(voice_controller.router, prefix="/api", tags=["voice-chat"])
app.include_router(fal_controller.router, prefix="/api", tags=["image"])
app.include_router(diary_controller.router, prefix="/api", tags=["diary"])
app.include_router(persona_controller.router, prefix="/api", tags=["persona"])
app.include_router(tone_controller.router, prefix="/api", tags=["tone"])
