import os
import uuid
import json
import asyncio
from difflib import SequenceMatcher
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.util.audio_utils import decode_and_resample_v2 as decode_and_resample, save_wav
from app.services.voice.stt_recorder import get_stt_recorder

router = APIRouter()
@router.websocket("/ws/pronunciation-test")
async def pronunciation_test(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()

    state = {"expected_script": None}
    pcm_buffer = bytearray()

    async def on_realtime(text: str):
        await ws.send_json({"type": "realtime", "text": text})

    async def on_full(full: str):
        await ws.send_json({"type": "fullSentence", "text": full})

        script = state["expected_script"] or ""
        ratio = SequenceMatcher(None, script, full).ratio()
        verdict = "OK" if ratio >= 0.8 else "RETRY"
        await ws.send_json({
            "type": "result",
            "accuracy": round(ratio, 3),
            "verdict": verdict
        })

        if verdict == "OK":
            save_dir = "/home/keem/refer"
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{uuid.uuid4().hex}.wav"
            path = os.path.join(save_dir, fname)
            save_wav(bytes(pcm_buffer), sample_rate=24000, path=path)
            await ws.send_json({"type": "saved", "path": path})
            await ws.close()

    cfg = {
        'device': 'cuda',
        'spinner': False,
        'use_microphone': False,
        'model': 'large-v3',
        'language': 'ko',
        'silero_sensitivity': 0.6,
        'webrtc_sensitivity': 1,
        'post_speech_silence_duration': 1,
        'min_length_of_recording': 0,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.05,
        'use_main_model_for_realtime': True,
    }

    recorder = get_stt_recorder(
        cfg,
        on_realtime=lambda t: asyncio.run_coroutine_threadsafe(on_realtime(t), loop),
        on_full_sentence=lambda s: asyncio.run_coroutine_threadsafe(on_full(s), loop)
    )

    try:
        while True:
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                print("[WebSocket] 클라이언트 연결 끊김")
                break
            except RuntimeError as e:
                print(f"[WebSocket] 수신 중 에러: {e}")
                break

            if msg.get("type") == "websocket.receive" and "text" in msg:
                data = json.loads(msg["text"])
                if "script" in data:
                    state["expected_script"] = data["script"]

            elif msg.get("type") == "websocket.receive" and "bytes" in msg:
                raw = msg["bytes"]
                header_len = int.from_bytes(raw[:4], 'little')
                meta = json.loads(raw[4:4+header_len].decode())
                pcm_chunk = raw[4+header_len:]
                pcm = decode_and_resample(
                    pcm_chunk,
                    meta.get('sampleRate', 16000),
                    24000
                )
                pcm_buffer.extend(pcm)
                recorder.feed_audio(pcm)

    finally:
        try:
            recorder.reset()  # STT 버퍼 비움 (내부 지원 여부 따라 수정)
        except Exception as e:
            print(f"[STT] recorder 종료 중 에러: {e}")
        pcm_buffer.clear()
        print("[WebSocket] 세션 종료 및 버퍼 초기화 완료")
