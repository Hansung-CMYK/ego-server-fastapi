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
import numpy as np
import time

logger = logging.getLogger(__name__)

REALTIME_STT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../modules/RealtimeSTT")
)
if REALTIME_STT_PATH not in sys.path:
    sys.path.insert(0, REALTIME_STT_PATH)

from RealtimeSTT.audio_recorder import AudioToTextRecorder
from app.util.audio_utils import decode_and_resample
from .tts_buffer import TTSBuffer
from app.services.chat.chat_service import chat_stream
from app.services.session_config import SessionConfig


from app.services.kafka.kafka_handler import get_producer, RESPONSE_AI_TOPIC, RESPONSE_CLIENT_TOPIC, ChatMessage, ContentType

async def produce_message(sentence: str, config: SessionConfig, topic: any):
    if not sentence.strip():
        return

    message = ChatMessage(
        chatRoomId=config.chat_room_id,
        from_=config.user_id,       # STT 결과: 사용자
        to=config.ego_id,       
        content=sentence,
        contentType=ContentType.TEXT,
        mcpEnabled=False
    ) if topic == RESPONSE_CLIENT_TOPIC else ChatMessage(
        chatRoomId=config.chat_room_id,
        from_=config.ego_id,       # TTS 결과: AI
        to=config.user_id,         
        content=sentence,
        contentType=ContentType.TEXT,
        mcpEnabled=False
    ) 

    try:
        producer = get_producer()
        await producer.send_and_wait(
            topic,
            key=message.from_,
            value=message,
        )
    except Exception as e:
        logger.exception(f"Kafka produce failed: {e}")

def send_sentence_from_sync(sentence: str, config: SessionConfig, topic : any, loop: asyncio.AbstractEventLoop):
    asyncio.run_coroutine_threadsafe(produce_message(sentence, config, topic), loop)


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
            'silero_sensitivity': 0.6,
            'webrtc_sensitivity': 1,
            'post_speech_silence_duration': 0.7,
            'min_length_of_recording': 0,
            'min_gap_between_recordings': 0,
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0,
            'use_main_model_for_realtime': True,
            'compute_type': 'float16'
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
        send_sentence_from_sync(full, self.config, RESPONSE_CLIENT_TOPIC, self.loop)
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

            final_pcm = None
            sr_of_final = None

            logger.warning(clean)
            start_time = time.monotonic()
            pcm = bytearray()

            from app.services.voice.tts_model_registry import get_tts_pipeline
            import gpt_sovits_api

            speaker = gpt_sovits_api.speaker_list.get(self.config.spk)
            tts_pipe = get_tts_pipeline(self.config.spk)
            tts_pipe.set_ref_audio(speaker.default_refer.path)

            gen = tts_pipe.run({
                "text": clean,
                "text_lang": "ko",
                "ref_audio_path": speaker.default_refer.path,
                "prompt_text": speaker.default_refer.text,
                "prompt_lang": speaker.default_refer.lang,
                "sample_steps": 16,
                "speed_factor": 1.0,
                "sample_steps": 4
            })

            for sr, chunk in gen:
                sr_of_final = sr
                if cancel_event.is_set():
                    return
                pcm.extend(chunk)
                elapsed = time.monotonic() - start_time

            if final_pcm is None:
                final_pcm = pcm

            from io import BytesIO
            import numpy as np

            pcm_np = np.frombuffer(bytes(final_pcm), dtype=np.int16)

            wav_buf = BytesIO()
            wav_buf = gpt_sovits_api.pack_wav(wav_buf, pcm_np, sr_of_final or self.sample_rate)
            wav_bytes = wav_buf.getvalue()

            payload = {
                "type": "audio_chunk",
                "index": tts_index,
                "audio_base64": base64.b64encode(wav_bytes).decode("ascii")
            }
            self._send_payload(payload)

        content = ""
        tts_buffer = TTSBuffer(send_tts)
        for chunk in chat_stream(prompt, self.config):
            if cancel_event.is_set():
                return
            self._send(type='response_chunk', text=chunk)
            tts_buffer.feed(chunk)
            content  += chunk
        
        send_sentence_from_sync(content, self.config, RESPONSE_AI_TOPIC, self.loop)

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
