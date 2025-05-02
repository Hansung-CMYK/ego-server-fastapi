import os
import sys
import asyncio
import threading
import json
import uuid
from app.services.tts_infer import get_tts_wav
from app.services.ollama_service import chat_stream

REALTIME_STT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../modules/RealtimeSTT"))
if REALTIME_STT_PATH not in sys.path:
    sys.path.insert(0, REALTIME_STT_PATH)

import faster_whisper
from faster_whisper import WhisperModel as _OriginalWhisperModel

__whisper_model_cache = {}
def _shared_whisper_model(model_size_or_path, device="cuda", compute_type="default",
                          device_index=None, download_root=None):
    key = (model_size_or_path, device, compute_type, device_index, download_root)
    if key not in __whisper_model_cache:
        __whisper_model_cache[key] = _OriginalWhisperModel(
            model_size_or_path=model_size_or_path,
            device=device,
            compute_type=compute_type,
            device_index=device_index,
            download_root=download_root,
        )
    return __whisper_model_cache[key]

faster_whisper.WhisperModel = _shared_whisper_model

from RealtimeSTT.audio_recorder import AudioToTextRecorder
from app.util.audio_utils import decode_and_resample

class VoiceChatSession:
    def __init__(self, websocket):
        self.id = str(uuid.uuid4())
        self.ws = websocket
        self.loop = asyncio.get_running_loop()
        self.recorder = AudioToTextRecorder(
            **self._recorder_config(),
            on_realtime_transcription_stabilized=self.on_text_detected
        )
        self.running = True
        threading.Thread(target=self.run_recorder, daemon=True).start()

    def _recorder_config(self):
        return {
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
        }

    def handle_audio(self, msg: bytes):
        ml = int.from_bytes(msg[:4], 'little')
        meta = json.loads(msg[4:4+ml].decode())
        pcm_chunk = msg[4+ml:]
        pcm = decode_and_resample(pcm_chunk, meta.get('sampleRate', 16000), 16000)
        self.recorder.feed_audio(pcm)

    def on_text_detected(self, text):
        asyncio.run_coroutine_threadsafe(
            self.ws.send_text(json.dumps({'type': 'realtime', 'text': text})),
            self.loop
        )

    def run_recorder(self):
        ref_wav_path = "/home/keem/sample.wav"
        prompt_language = "ja"
        text_language = "ko"

        while self.running:
            full = self.recorder.text()
            if not full:
                continue

            asyncio.run_coroutine_threadsafe(
                self.ws.send_text(json.dumps({'type': 'fullSentence', 'text': full})),
                self.loop
            )

            response_chunks = []
            for chunk in chat_stream(full):
                response_chunks.append(chunk)
                asyncio.run_coroutine_threadsafe(
                    self.ws.send_text(json.dumps({'type': 'response_chunk', 'text': chunk})),
                    self.loop
                )

            asyncio.run_coroutine_threadsafe(
                self.ws.send_text(json.dumps({'type': 'response_done'})),
                self.loop
            )

            full_response = ''.join(response_chunks)
            pcm_bytes = bytearray()
            for audio_chunk in get_tts_wav(ref_wav_path, "...", prompt_language, full_response, text_language):
                pcm_bytes.extend(audio_chunk)

            asyncio.run_coroutine_threadsafe(
                self.ws.send_bytes(pcm_bytes),
                self.loop
            )

    def stop(self):
        self.running = False


class VoiceChatSessionManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self, ws):
        session = VoiceChatSession(ws)
        self.sessions[session.id] = session
        return session

    def remove_session(self, sid):
        if sid in self.sessions:
            self.sessions[sid].stop()
            del self.sessions[sid]

