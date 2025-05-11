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
from .tts_buffer import TTSBuffer
from app.services.ollama_service import chat_stream
from .session_config import SessionConfig

class VoiceChatHandler:
    def __init__(self, websocket, config: SessionConfig):
        self.id = str(uuid.uuid4())
        self.ws = websocket
        self.loop = asyncio.get_running_loop()
        self.running = True
        self.cancel_event = threading.Event()
        self.llm_thread = None

        self.config = config

        self._init_recorder()  # 녹음기 초기화 및 쓰레드 시작

    def _init_recorder(self):
        config = self._recorder_config()
        config['on_realtime_transcription_stabilized'] = self._on_realtime
        self.recorder = AudioToTextRecorder(**config)
        threading.Thread(target=self._recorder_loop, daemon=True).start()

    def _recorder_config(self) -> dict:
        return {
            'spinner': False,
            'use_microphone': False,
            'model': 'large-v3',
            'language': 'ko',
            'silero_sensitivity': 0.6,
            'webrtc_sensitivity': 1,
            'post_speech_silence_duration': 0.4,
            'min_length_of_recording': 0,
            'min_gap_between_recordings': 0,
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0,
            'realtime_model_type': 'large-v3-turbo',
        }

    def handle_audio(self, msg: bytes):
        # 클라이언트 오디오 바이트 디코딩 후 녹음기에 전달
        header_len = int.from_bytes(msg[:4], 'little')
        meta = json.loads(msg[4:4 + header_len].decode())
        pcm_chunk = msg[4 + header_len:]
        pcm = decode_and_resample(pcm_chunk, meta.get('sampleRate', 16000), 16000)
        self.recorder.feed_audio(pcm)

    def _on_realtime(self, text: str):
        # 부분 인식 결과 전송 및 기존 TTS/응답 중단 신호
        self._send(type='realtime', text=text)
        self._send(type='cancel_audio')

    def _recorder_loop(self):
        # 안정된 문장 완성 대기 루프
        while self.running:
            full = self.recorder.text()
            if not full:
                continue
            self._process_full_sentence(full)

    def _process_full_sentence(self, full: str):
        # 새 문장 수신 시 기존 작업 취소 후 다시 생성 시작
        self._cancel_current()
        self._send(type='fullSentence', text=full)
        self._start_llm_tts(full)

    def _cancel_current(self):
        # 현재 진행 중인 프로세스 중단
        self.cancel_event.set()
        if self.llm_thread and self.llm_thread.is_alive():
            self.llm_thread.join()
        self.cancel_event = threading.Event()
        self._send(type='cancel_audio')

    def _start_llm_tts(self, full: str):
        # LLM 스트리밍, TTS 백그라운드 실행
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
            # 텍스트 정제 후 TTS 생성
            clean = self._clean_text(sentence)
            if not clean:
                return
            pcm = self._generate_tts(clean, cancel_event)
            if cancel_event.is_set():
                return
            # 생성된 오디오 청크 전송
            payload = {
                'type': 'audio_chunk',
                'index': tts_index,
                'audio_base64': base64.b64encode(pcm).decode('ascii')
            }
            self._send_payload(payload)
            tts_index += 1

        # LLM 응답 스트리밍 + TTS 버퍼 처리
        tts_buffer = TTSBuffer(send_tts)
        for chunk in chat_stream(prompt, self.config):
            if cancel_event.is_set():
                return
            self._send(type='response_chunk', text=chunk)
            tts_buffer.feed(chunk)

        # 완료 신호 및 남은 버퍼 flush
        if not cancel_event.is_set():
            self._send(type='response_done')
            tts_buffer.flush()

    def _clean_text(self, text: str) -> str:
        # 마크다운·특수문자 제거 및 공백 정규화
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'(__|\*\*|\*|`|~~|!\[.*?\]\(.*?\)|\[.*?\]\(.*?\))', '', text)
        text = text.replace('\n', '').replace('\r', '').replace('\t', '')
        text = re.sub(r'(?<=\d)[.,?!:]', '', text)
        text = re.sub(r'[^\uAC00-\uD7A3\u3131-\u318F0-9A-Za-z,.!? ]+', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _generate_tts(self, text: str, cancel_event: threading.Event) -> bytearray:
        # GPT-SoVITS를 통해 WAV 청크 생성
        gsv = importlib.import_module("gpt_sovits_api")
        pcm = bytearray()
        for chunk in gsv.get_tts_wav(
            ref_wav_path="/home/keem/sample.wav",
            prompt_text="なるべく、教師との無駄なやり取りを発生させたくないですもんね。",
            prompt_language="ja",
            text=text,
            text_language="ko"
        ):
            if cancel_event.is_set():
                return bytearray()
            pcm.extend(chunk)
        return pcm

    def _send(self, **kwargs):
        # JSON 메시지 전송
        asyncio.run_coroutine_threadsafe(
            self.ws.send_text(json.dumps(kwargs)),
            self.loop
        )

    def _send_payload(self, payload: dict):
        # 페이로드 전송
        asyncio.run_coroutine_threadsafe(
            self.ws.send_text(json.dumps(payload)),
            self.loop
        )

    def stop(self):
        # 세션 종료 시 모든 스레드 중단
        self.running = False
        self.cancel_event.set()
        if self.llm_thread and self.llm_thread.is_alive():
            self.llm_thread.join()
