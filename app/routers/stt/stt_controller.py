import asyncio
import json
import math
import os
import uuid
import wave
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from scipy.signal import resample_poly

from config.keem.voice.stt_recorder import get_stt_recorder, release_stt_recorder

router = APIRouter()

TARGET_SR = 16_000
SAVE_DIR = str(Path.home() / "refer")
os.makedirs(SAVE_DIR, exist_ok=True)

def _decode_resample_le(pcm_bytes: bytes, src_sr: int, tgt_sr: int = TARGET_SR) -> bytes:
    audio = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    if src_sr != tgt_sr:
        g = math.gcd(src_sr, tgt_sr)
        audio = resample_poly(audio, tgt_sr // g, src_sr // g)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype("<i2").tobytes()

def _save_wav(pcm: bytes, sr: int, path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    print(f"[PRINT] WAV saved → {path}")

@router.websocket("/ws/pronunciation-_test")
async def pronunciation_test(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()
    print("[PRINT] ▶ WS connection accepted")

    expected_script = None
    pcm_buffer = bytearray()
    src_sr = None

    async def on_realtime(text: str):
        await ws.send_json({"type":"realtime","text":text})
        print(f"[PRINT] ▶ realtime: {text}")

    async def on_full(full: str):
        # fullSentence + result 전송
        await ws.send_json({"type":"fullSentence","text":full})
        ratio = SequenceMatcher(None, expected_script or "", full).ratio()
        verdict = "OK" if ratio >= 0.8 else "RETRY"
        await ws.send_json({"type":"result","accuracy":round(ratio,3),"verdict":verdict})
        print(f"[PRINT] ▶ fullSentence: '{full}' ratio={ratio:.3f} verdict={verdict}")

        # OK면 저장만, 세션은 유지
        if verdict == "OK":
            fname = f"{uuid.uuid4().hex}.wav"
            path = os.path.join(SAVE_DIR, fname)
            _save_wav(bytes(pcm_buffer), TARGET_SR, path)
            await ws.send_json({"type":"saved","path":path})
            print("[PRINT] ▶ audio saved; session remains open")

    cfg = {
        "device":"cuda","spinner":False,"use_microphone":False,
        "model":"large-v3","language":"ko",
        "silero_sensitivity":0.6,"webrtc_sensitivity":1,
        "post_speech_silence_duration":0.3,
        "min_length_of_recording":0,"min_gap_between_recordings":0,
        "enable_realtime_transcription":True,"realtime_processing_pause":0.05,
        "use_main_model_for_realtime":True,
    }

    recorder = get_stt_recorder(
        cfg,
        on_realtime=lambda t: asyncio.run_coroutine_threadsafe(on_realtime(t), loop),
        on_full_sentence=lambda s: asyncio.run_coroutine_threadsafe(on_full(s), loop),
    )
    if recorder is None:
        await ws.send_json({"type":"error","message":"STT 엔진이 사용 중입니다."})
        await ws.close()
        return
    print("[PRINT] ▶ STT recorder allocated")

    try:
        while True:
            try:
                msg = await ws.receive()
            except (WebSocketDisconnect, RuntimeError):
                print("[PRINT] ▶ WS disconnected or closed")
                break

            if msg.get("type") != "websocket.receive":
                continue

            # 텍스트 메시지 처리 (script 또는 eos)
            if "text" in msg:
                data = json.loads(msg["text"])

                if data.get("eos"):
                    print("[PRINT] ▶ Received EOS — flushing buffer")
                    # 남은 버퍼 강제 처리
                    try:
                        recorder.stop()
                    except:
                        pass
                    # on_full 로 강제 전달 (빈 텍스트도 내부 버퍼로)
                    await on_full("")
                    pcm_buffer.clear()
                    # 재할당 없이 동일 recorder 사용 복원
                    continue

                if "script" in data:
                    expected_script = data["script"]
                    print(f"[PRINT] ▶ script set to '{expected_script}'")
                continue

            # 오디오 바이트 처리
            if "bytes" in msg:
                raw = msg["bytes"]
                hdr_len = int.from_bytes(raw[:4], 'little')
                meta = json.loads(raw[4:4+hdr_len].decode())
                block = raw[4+hdr_len:]
                sr = int(meta.get("sampleRate", TARGET_SR))
                if src_sr is None:
                    src_sr = sr
                    print(f"[PRINT] ▶ first packet sampleRate={src_sr}")

                pcm16 = _decode_resample_le(block, src_sr, TARGET_SR)
                pcm_buffer.extend(pcm16)
                recorder.feed_audio(pcm16)
                print(f"[PRINT] ▶ feed {len(pcm16)} bytes @16k")
    except WebSocketDisconnect:
        print("[PRINT] ▶ WS disconnected by client")
    finally:
        # 세션 종료 시 리소스 해제
        try: recorder.stop()
        except: pass
        release_stt_recorder()
        print("[PRINT] ▶ STT recorder released")
        print("[PRINT] ▶ WS session ended")