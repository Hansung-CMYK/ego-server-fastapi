import uuid
import json
import threading
import asyncio
import re
import base64
import emoji
import logging
import numpy as np
import time

from app.util.audio_utils import decode_and_resample_v2 as decode_and_resample
from .tts_buffer import TTSBuffer
from app.services.chat.chat_service import chat_stream
from app.services.session_config import SessionConfig
from app.services.diary.kobert_handler import extract_emotions

from app.services.voice.stt_recorder import get_stt_recorder, release_stt_recorder

from app.services.kafka.kafka_handler import (
    wait_until_kafka_ready,
    get_producer,
    RESPONSE_AI_TOPIC,
    RESPONSE_CLIENT_TOPIC,
    ChatMessage,
    ContentType
)

from app.logger.logger import logger

async def produce_message(sentence: str, config: SessionConfig, topic: str):
    await wait_until_kafka_ready()
    if not sentence.strip():
        return

    message = ChatMessage(
        chatRoomId=config.chat_room_id,
        from_=config.user_id if topic == RESPONSE_CLIENT_TOPIC else config.ego_id,
        to=config.ego_id if topic == RESPONSE_CLIENT_TOPIC else config.user_id,
        content=sentence,
        contentType=ContentType.TEXT,
        mcpEnabled=False
    )

    try:
        producer = get_producer()
        await producer.send_and_wait(topic, key=message.from_, value=message)
    except Exception as e:
        logger.exception(f"Kafka produce failed: {e}")

def send_sentence_from_sync(sentence: str, config: SessionConfig, topic: str, loop: asyncio.AbstractEventLoop):
    asyncio.run_coroutine_threadsafe(
        produce_message(sentence, config, topic),
        loop
    )

class VoiceChatHandler:
    INACTIVITY_TIMEOUT = 10.0

    def __init__(self, websocket, config: SessionConfig):
        self.id = str(uuid.uuid4())
        self.ws = websocket
        self.loop = asyncio.get_running_loop()
        self.config = config

        self.running = True
        self.cancel_event = threading.Event()
        self.llm_thread = None

        self.sample_rate = 24000
        self._last_audio_time = time.monotonic()
        self._has_stt_lock = False

        try:
            self._init_recorder()
            self._has_stt_lock = True
        except RuntimeError:
            asyncio.run_coroutine_threadsafe(
                self.ws.send_text(json.dumps({
                    "type": "error",
                    "message": "STT 엔진이 사용 중입니다. 잠시 후 다시 시도해주세요."
                })),
                self.loop
            )
            raise

        self._start_inactivity_watchdog()

    def _recorder_config(self) -> dict:
        return {
            'device': 'cuda',
            'spinner': False,
            'use_microphone': False,
            'model': 'large-v3',
            'language': 'ko',
            'silero_sensitivity': 0.4,
            'webrtc_sensitivity': 1,
            'post_speech_silence_duration': 0.6,
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0.1,
            'use_main_model_for_realtime': True,
        }

    def _init_recorder(self):
        cfg = self._recorder_config()
        self.recorder = get_stt_recorder(
            cfg,
            on_realtime=self._on_realtime,
            on_full_sentence=self._process_full_sentence
        )

        if self.recorder is None:
            logger.warning(f"[{self.id}] STT 엔진 할당 실패 — 이미 사용 중")
            asyncio.run_coroutine_threadsafe(
                self.ws.send_text(json.dumps({
                    "type": "error",
                    "message": "현재 다른 세션이 STT를 사용 중입니다. 잠시 후 다시 시도해주세요."
                })),
                self.loop
            )
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
            self.running = False
            return

    def _start_inactivity_watchdog(self):
        def watchdog():
            while self.running:
                elapsed = time.monotonic() - self._last_audio_time
                if elapsed > self.INACTIVITY_TIMEOUT:
                    logger.info(f"[{self.id}] 무입력 {elapsed:.1f}s 경과, 세션 종료")
                    self.stop()
                    break
                time.sleep(5.0)
        t = threading.Thread(target=watchdog, daemon=True)
        t.start()

    def handle_audio(self, msg: bytes):
        logger.info(msg=f"\n\nPOST: api/v1/ws/voice-chat [handle_audio]")

        # 오디오 수신 시 타이머 갱신
        self._last_audio_time = time.monotonic()

        header_len = int.from_bytes(msg[:4], 'little')
        meta = json.loads(msg[4:4+header_len].decode())
        pcm_chunk = msg[4+header_len:]
        pcm = decode_and_resample(pcm_chunk, meta.get('sampleRate', 16000), self.sample_rate)
        self.recorder.feed_audio(pcm)

    def _on_realtime(self, text: str):
        # 실시간 텍스트 수신 시
        self._last_audio_time = time.monotonic()
        self._send(type='realtime', text=text)
        self._send(type='cancel_audio')

    def _process_full_sentence(self, full: str):
        logger.info(msg=f"\n\nPOST: api/v1/ws/voice-chat [_process_full_sentence]")
        # 완전 문장 수신 시
        self._last_audio_time = time.monotonic()
        self._cancel_current()
        self._send(type='fullSentence', text=full)
        send_sentence_from_sync(full, self.config, RESPONSE_CLIENT_TOPIC, self.loop)
        self._start_llm_tts(full)

    def _cancel_current(self):
        logger.info(msg=f"\n\nPOST: api/v1/ws/voice-chat [_cancel_current]")
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

            from app.services.voice.tts_model_registry import get_tts_pipeline
            import gpt_sovits_api

            speaker = gpt_sovits_api.speaker_list.get(self.config.spk)
            tts_pipe = get_tts_pipeline(self.config.spk)
            tts_pipe.set_ref_audio(speaker.default_refer.path)

            pcm = bytearray()
            sr_of_final = None

            gen = tts_pipe.run({
                "text": clean,
                "text_lang": "ko",
                "ref_audio_path": speaker.default_refer.path,
                "prompt_text": speaker.default_refer.text,
                "prompt_lang": speaker.default_refer.lang,
                "sample_steps": 16,
                "speed_factor": 1.0,
            })

            for sr, chunk in gen:
                if cancel_event.is_set():
                    return
                sr_of_final = sr
                pcm.extend(chunk)

            from io import BytesIO
            wav_buf = BytesIO()
            wav_buf = gpt_sovits_api.pack_wav(
                wav_buf,
                np.frombuffer(bytes(pcm), dtype=np.int16),
                sr_of_final or self.sample_rate
            )
            wav_bytes = wav_buf.getvalue()

            payload = {
                "type": "audio_chunk",
                "index": tts_index,
                "audio_base64": base64.b64encode(wav_bytes).decode("ascii")
            }
            self._send_payload(payload)
            tts_index += 1

        content = ""
        tts_buffer = TTSBuffer(send_tts)
        for chunk in chat_stream(prompt, self.config):
            if cancel_event.is_set():
                return
            self._send(type='response_chunk', text=chunk)
            tts_buffer.feed(chunk)
            content += chunk

        send_sentence_from_sync(content, self.config, RESPONSE_AI_TOPIC, self.loop)
        self._send_emotion(content)

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

    def _send_emotion(self, text: str):
        emotion = extract_emotions([text], alpha=0.5, top_k=1)
        payload = {
            "type": "emotion",
            "emotion": emotion
        }
        asyncio.run_coroutine_threadsafe(
            self.ws.send_text(json.dumps(payload)),
            self.loop
        )

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
        # 세션 종료 시 호출
        if not self.running:
            return
        self.running = False
        self.cancel_event.set()
        if self.llm_thread and self.llm_thread.is_alive():
            self.llm_thread.join()

        # STT recorder 정지
        try:
            self.recorder.stop()
        except Exception:
            pass

        # 세마포어 해제
        if self._has_stt_lock:
            release_stt_recorder()
            self._has_stt_lock = False

        logger.info(f"[{self.id}] VoiceChatHandler 리소스 정리 완료")
