import os
import sys
import threading
import asyncio
import json
import logging

import numpy as np
from scipy.signal import resample
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODULES_ROOT = os.path.join(ROOT, "modules")
if MODULES_ROOT not in sys.path:
    sys.path.insert(0, MODULES_ROOT)

from RealtimeSTT.RealtimeSTT.audio_recorder import AudioToTextRecorder
from app.services.ollama_service import chat_stream

router = APIRouter()
logging.basicConfig(level=logging.INFO)

recorder = None
recorder_ready = threading.Event()
client_ws = None
main_loop = None
is_running = True

async def send_to_client(obj: dict):
    global client_ws
    if client_ws:
        try:
            await client_ws.send_text(json.dumps(obj))
        except:
            client_ws = None

def text_detected(text: str):
    global main_loop
    if main_loop:
        asyncio.run_coroutine_threadsafe(
            send_to_client({'type': 'realtime', 'text': text}),
            main_loop
        )

def run_recorder():
    global recorder, main_loop, is_running
    recorder = AudioToTextRecorder(**recorder_config)
    recorder_ready.set()

    while is_running:
        full = recorder.text()
        if full and main_loop:
            asyncio.run_coroutine_threadsafe(
                send_to_client({'type': 'fullSentence', 'text': full}),
                main_loop
            )

            for chunk in chat_stream(full):
                logging.info(f"→ sending response_chunk: {chunk!r}")
                asyncio.run_coroutine_threadsafe(
                    send_to_client({'type': 'response_chunk', 'text': chunk}),
                    main_loop
                )

            asyncio.run_coroutine_threadsafe(
                send_to_client({'type': 'response_done'}),
                main_loop
            )

def decode_and_resample(data: bytes, sr: int, tr: int) -> bytes:
    arr = np.frombuffer(data, np.int16)
    tgt = int(len(arr) * tr / sr)
    out = resample(arr, tgt)
    return out.astype(np.int16).tobytes()

recorder_config = {
    'spinner': False,
    'use_microphone': False,
    'model': 'large-v3',
    'language': 'ko',
    'silero_sensitivity': 0.4,
    'webrtc_sensitivity': 2,
    'post_speech_silence_duration': 0.4,
    'min_length_of_recording': 0,
    'min_gap_between_recordings': 0,
    'enable_realtime_transcription': True,
    'realtime_processing_pause': 0,
    'realtime_model_type': 'large-v3-turbo',
    'on_realtime_transcription_stabilized': text_detected,
}

@router.websocket("/ws/voice-chat")
async def ws_vc(ws: WebSocket):
    global client_ws, main_loop, is_running
    await ws.accept()
    client_ws = ws

    if not recorder_ready.is_set():
        main_loop = asyncio.get_running_loop()
        t = threading.Thread(target=run_recorder, daemon=True)
        t.start()
        recorder_ready.wait()

    try:
        while True:
            msg = await ws.receive_bytes()
            ml = int.from_bytes(msg[:4], 'little')
            meta = json.loads(msg[4:4+ml].decode())
            pcm_chunk = msg[4+ml:]
            pcm = decode_and_resample(pcm_chunk, meta.get('sampleRate', 16000), 16000)
            recorder.feed_audio(pcm)
    except WebSocketDisconnect:
        client_ws = None
        is_running = False

