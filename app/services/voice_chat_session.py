import os
import sys
import uuid
import json
import threading
import importlib
import asyncio
import re
import base64
import emoji

REALTIME_STT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../modules/RealtimeSTT")
)
if REALTIME_STT_PATH not in sys.path:
    sys.path.insert(0, REALTIME_STT_PATH)

from RealtimeSTT.audio_recorder import AudioToTextRecorder
from app.util.audio_utils import decode_and_resample
from .whisper_model import get_shared_whisper_model
from .tts_buffer import TTSBuffer
from app.services.ollama_service import chat_stream

class VoiceChatSession:
    def __init__(self, websocket):
        self.id = str(uuid.uuid4())
        self.ws = websocket
        self.loop = asyncio.get_running_loop()
        self.running = True
        self.cancel_event = threading.Event()
        self.llm_thread = None
        self.recorder = self._initialize_recorder()
        threading.Thread(target=self.run_recorder, daemon=True).start()

    def _initialize_recorder(self):
        config = self._recorder_config()
        return AudioToTextRecorder(
            **config,
            on_realtime_transcription_stabilized=self.on_text_detected,
            on_vad_start=self.halt_current_process
        )

    def _recorder_config(self):
        return {
            'spinner': False,
            'use_microphone': False,
            'model': 'large-v3',
            'language': 'ko',
            'silero_sensitivity': 0.6,
            'webrtc_sensitivity': 2,
            'pre_recording_buffer_duration': 0.4,
            'post_speech_silence_duration': 0.4,
            'min_length_of_recording': 1.0,
            'min_gap_between_recordings': 0,
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0,
            'realtime_model_type': 'large-v3-turbo',
        }

    def handle_audio(self, msg: bytes):
        meta, pcm = self._extract_audio_data(msg)
        self.recorder.feed_audio(pcm)

    def _extract_audio_data(self, msg: bytes):
        ml = int.from_bytes(msg[:4], 'little')
        meta = json.loads(msg[4:4+ml].decode())
        pcm_chunk = msg[4+ml:]
        pcm = decode_and_resample(pcm_chunk, meta.get('sampleRate', 16000), 16000)
        return meta, pcm

    def on_text_detected(self, text: str):
        self._send_to_client({'type': 'realtime', 'text': text})

    def _send_to_client(self, data):
        asyncio.run_coroutine_threadsafe(
            self.ws.send_text(json.dumps(data)),
            self.loop
        )

    def run_recorder(self):
        while self.running:
            full = self.recorder.text()
            if not full:
                continue
            self.halt_current_process()
            self._send_to_client({'type': 'fullSentence', 'text': full})
            self.llm_thread = threading.Thread(
                target=self._llm_tts_worker,
                args=(full, self.cancel_event),
                daemon=True
            )
            self.llm_thread.start()

    def _llm_tts_worker(self, full: str, cancel_event: threading.Event):
        tts_buffer = TTSBuffer(self._send_tts)
        for chunk in chat_stream(full):
            if cancel_event.is_set():
                return
            self._send_to_client({'type': 'response_chunk', 'text': chunk})
            tts_buffer.feed(chunk)
        if not cancel_event.is_set():
            self._send_to_client({'type': 'response_done'})
            tts_buffer.flush()

    def _send_tts(self, sentence: str):
        text = self._clean_text(sentence)
        if not text:
            return
        gsv = importlib.import_module("gpt_sovits_api")
        pcm = bytearray()
        for chunk in gsv.get_tts_wav(
            ref_wav_path="/home/keem/sample.wav",
            prompt_text="なるべく、教師との無駄なやり取りを発生させたくないですもんね。",
            prompt_language="ja",
            text=text,
            text_language="ko"
        ):
            pcm.extend(chunk)
        payload = {
            "type": "audio_chunk",
            "audio_base64": base64.b64encode(pcm).decode('ascii')
        }
        self._send_to_client(payload)

    def _clean_text(self, text: str) -> str:
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'(__|\*\*|\*|`|~~|!\[.*?\]\(.*?\)|\[.*?\]\(.*?\))', '', text)
        text = re.sub(r'[\s]+', ' ', text.strip())
        return text

    def stop(self):
        self.running = False
        self.cancel_event.set()
        if self.llm_thread and self.llm_thread.is_alive():
            self.llm_thread.join()

    def halt_current_process(self):
        self.cancel_event.set()
        if self.llm_thread and self.llm_thread.is_alive():
            self.llm_thread.join()
        self._send_to_client({'type': 'cancel_audio'})
