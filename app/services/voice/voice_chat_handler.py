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
import logging
import torch
import numpy as np
import torchaudio
import time

logger = logging.getLogger(__name__)

silero_model, silero_utils = torch.hub.load(
    'snakers4/silero-vad', 'silero_vad', force_reload=False
)
(get_speech_timestamps, _, read_audio, *_ ) = silero_utils

REALTIME_STT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../modules/RealtimeSTT")
)
if REALTIME_STT_PATH not in sys.path:
    sys.path.insert(0, REALTIME_STT_PATH)

from RealtimeSTT.audio_recorder import AudioToTextRecorder
from app.util.audio_utils import decode_and_resample
from .tts_buffer import TTSBuffer
from app.services.chatting.chat_service import chat_stream
from app.services.session_config import SessionConfig

CHAR_TO_SEC = 0.05
TIMEOUT_MARGIN = 2.5

def speech_ratio(pcm: bytes, sample_rate: int = 24000) -> float:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    tensor = torch.from_numpy(samples)
    sr = sample_rate
    if sr != 16000:
        tensor = tensor.unsqueeze(0)
        tensor = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(tensor).squeeze(0)
        sr = 16000
    timestamps = get_speech_timestamps(tensor, silero_model, sampling_rate=sr)
    if not timestamps:
        return 0.0
    voiced = sum(item['end'] - item['start'] for item in timestamps)
    return voiced / tensor.shape[0]

class VoiceChatHandler:
    def __init__(self, websocket, config: SessionConfig):
        self.id = str(uuid.uuid4())
        self.ws = websocket
        self.loop = asyncio.get_running_loop()
        self.running = True
        self.cancel_event = threading.Event()
        self.llm_thread = None

        self.config = config
        self.vad_threshold = 0.2
        self.max_retries = 2
        self.sample_rate = 24000

        self._init_recorder()

    def _init_recorder(self):
        cfg = self._recorder_config()
        cfg['on_realtime_transcription_stabilized'] = self._on_realtime
        self.recorder = AudioToTextRecorder(**cfg)
        threading.Thread(target=self._recorder_loop, daemon=True).start()

    def _recorder_config(self) -> dict:
        return {
            'device': 'cuda',
            'spinner': False,
            'use_microphone': False,
            'model': 'large-v3',
            'language': 'ko',
            'silero_sensitivity': 0.5,
            'webrtc_sensitivity': 1,
            'post_speech_silence_duration': 0.7,
            'min_length_of_recording': 0,
            'min_gap_between_recordings': 0,
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0,
            'use_main_model_for_realtime': True,
            'compute_type': 'int8'
        }

    def handle_audio(self, msg: bytes):
        header_len = int.from_bytes(msg[:4], 'little')
        meta = json.loads(msg[4:4+header_len].decode())
        pcm_chunk = msg[4+header_len:]
        pcm = decode_and_resample(pcm_chunk, meta.get('sampleRate', 16000), self.sample_rate)
        self.recorder.feed_audio(pcm)

    def _on_realtime(self, text: str):
        self._send(type='realtime', text=text)
        self._send(type='cancel_audio')

    def _recorder_loop(self):
        while self.running:
            full = self.recorder.text()
            if full:
                self._process_full_sentence(full)

    def _process_full_sentence(self, full: str):
        self._cancel_current()
        self._send(type='fullSentence', text=full)
        self._start_llm_tts(full)

    def _cancel_current(self):
        self.cancel_event.set()
        if self.llm_thread and self.llm_thread.is_alive():
            self.llm_thread.join()
        self.cancel_event = threading.Event()
        self._send(type='cancel_audio')

    def _start_llm_tts(self, full: str):
        self.llm_thread = threading.Thread(
            target=self._llm_tts_worker,
            args=(full, self.cancel_event),
            daemon=True
        )
        self.llm_thread.start()

    def _llm_tts_worker(self, prompt: str, cancel_event: threading.Event):
        tts_index = 0

        def send_tts(sentence: str):
            nonlocal tts_index
            if cancel_event.is_set():
                return
            clean = self._clean_text(sentence)
            if not clean:
                return

            expected_time = len(clean) * CHAR_TO_SEC
            timeout = expected_time * TIMEOUT_MARGIN

            retries = 0
            final_pcm = bytearray()

            logger.warning(clean)
            while retries <= self.max_retries:
                start_time = time.monotonic()
                pcm = bytearray()

                gen = importlib.import_module("gpt_sovits_api").get_tts_wav(
                    ref_wav_path="/home/keem/sample.wav",
                    prompt_text="오늘 우리 집에서 치킨 먹고 갈래?",
                    prompt_language="ko",
                    text=clean,
                    text_language="ko",
                    sample_steps=8,
                    speed=1.2,
                    spk="karina"
                )
                for chunk in gen:
                    if cancel_event.is_set():
                        return
                    pcm.extend(chunk)
                    elapsed = time.monotonic() - start_time
                    if elapsed > timeout:
                        logger.error(
                            "Timeout mid-generation idx %d after %.2f>%.2fsec, retry %d/%d",
                            tts_index, elapsed, timeout, retries+1, self.max_retries
                        )
                        break
                else:
                    elapsed = time.monotonic() - start_time
                    ratio = speech_ratio(pcm, self.sample_rate)
                    if elapsed <= timeout and ratio >= self.vad_threshold:
                        final_pcm = pcm
                        break
                    logger.error(
                        "Retry idx %d: elapsed=%.2fsec, voiced_ratio=%.2f",
                        tts_index, elapsed, ratio
                    )
                retries += 1

            if not final_pcm:
                logger.error(
                    "Failed to generate valid TTS for idx %d after %d retries",
                    tts_index, self.max_retries
                )
                final_pcm = pcm

            payload = {
                'type': 'audio_chunk',
                'index': tts_index,
                'audio_base64': base64.b64encode(final_pcm).decode('ascii')
            }
            self._send_payload(payload)
            tts_index += 1

        tts_buffer = TTSBuffer(send_tts)
        for chunk in chat_stream(prompt, self.config):
            if cancel_event.is_set():
                return
            self._send(type='response_chunk', text=chunk)
            tts_buffer.feed(chunk)

        if not cancel_event.is_set():
            self._send(type='response_done')
            tts_buffer.flush()

    def _clean_text(self, text: str) -> str:
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'(__|\*\*|\*|`|~~|!\[.*?\]\(.*?\)|\[.*?\]\(.*?\))', '', text)
        text = text.replace('\n', '').replace('\r', '').replace('\t', '')
        text = re.sub(r'(?<=\d)[.,?!:]', '', text)
        text = re.sub(r'[^\uAC00-\uD7A3\u3131-\u318F0-9A-Za-z,.!? ]+', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _send(self, **kwargs):
        asyncio.run_coroutine_threadsafe(
            self.ws.send_text(json.dumps(kwargs)),
            self.loop
        )

    def _send_payload(self, payload: dict):
        asyncio.run_coroutine_threadsafe(
            self.ws.send_text(json.dumps(payload)),
            self.loop
        )

    def stop(self):
        self.running = False
        self.cancel_event.set()
        if self.llm_thread and self.llm_thread.is_alive():
            self.llm_thread.join()
        try:
            self.recorder.stop()
            self.recorder.join()
        except Exception:
            pass
