from fastapi import FastAPI
from app.services.tts_initializer import load_tts_model

app = FastAPI()

@app.on_event("startup")
async def load_models():
    load_tts_model(
        model_id="default",
        gpt_path="modules/GPT_SoVITS/GPT_SoVITS/pretrained_models/s1v3.ckpt",
        sovits_path="modules/GPT_SoVITS/GPT_SoVITS/pretrained_models/s2Gv3.pth"
    )
