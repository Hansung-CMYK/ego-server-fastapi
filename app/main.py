import os
import sys
import importlib.util

from fastapi import FastAPI, APIRouter
from app.api import chat_api, voice_chat_api
from app.services.tts_infer import ensure_init

from contextlib import asynccontextmanager

import app.services.kafka_handler as kh
import asyncio

import logging
logging.basicConfig(level=logging.ERROR)

here = os.path.dirname(__file__)
api_file_path = os.path.abspath(os.path.join(here, "../modules/GPT-SoVITS/api.py"))
gpt_sovits_root = os.path.dirname(api_file_path)
gpt_sovits_sub  = os.path.join(gpt_sovits_root, "GPT_SoVITS")

for path in (gpt_sovits_root, gpt_sovits_sub):
    if path not in sys.path:
        sys.path.insert(0, path)

spec = importlib.util.spec_from_file_location("gpt_sovits_api", api_file_path)
gpt_sovits_api = importlib.util.module_from_spec(spec)
sys.modules["gpt_sovits_api"] = gpt_sovits_api
spec.loader.exec_module(gpt_sovits_api)

def init_models():
    model_id    = "default"
    gpt_path    = "/path/to/gpt_weights"
    sovits_path = "/path/to/sovits_weights"
    ensure_init(model_id, gpt_path, sovits_path)

async def on_startup():
    await kh.init_kafka()
    asyncio.create_task(kh.consume_loop())
    return 

async def on_shutdown():
    await kh.shutdown_kafka()
    return 

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_models()
    await on_startup()
    
    yield
    await on_shutdown()
    

app = FastAPI(lifespan=lifespan)

tts_router = APIRouter(prefix="/tts", tags=["tts"])

for route in gpt_sovits_api.app.routes:
    if route.path == "/":
        continue
    tts_router.routes.append(route)

app.include_router(tts_router)

app.include_router(chat_api.router,    prefix="/api", tags=["chat"])
app.include_router(voice_chat_api.router, prefix="/api", tags=["voice-chat"])
