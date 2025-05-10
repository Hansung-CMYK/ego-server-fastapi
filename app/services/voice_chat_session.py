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

        self.recorder = AudioToTextRecorder(
            **self._recorder_config(),
            on_realtime_transcription_stabilized=self.on_text_detected,

        )
        threading.Thread(target=self.run_recorder, daemon=True).start()

    def _recorder_config(self):
        return {
            'spinner': False,
            'use_microphone': False,
            'model': 'large-v3',
            'language': 'ko',
            # 민감도 0 (least sensitive) to 1 (most sensitive)
            'silero_sensitivity': 0.6,
            # 민감도 0 (most sensitive) to 3 (least sensitive)
            'webrtc_sensitivity': 2,
            # 사전 녹음 대기 시간
            "pre_recording_buffer_duration": 0.4,
            # 녹음 끝낼 대기 시간
            'post_speech_silence_duration': 0.4,
            # 최소 녹음 시간
            'min_length_of_recording': 1.0,
            # 녹음 끝난 이후 대기 시간
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

    def on_text_detected(self, text: str):
        asyncio.run_coroutine_threadsafe(
            self.ws.send_text(json.dumps({'type': 'realtime', 'text': text})),
            self.loop
        )

    def run_recorder(self):
        """
            RealtimeSTT 메인 로직

            1. transcribe된 텍스트 가져옴
            2. 이전에 생성중인 프로세스 중지 
            3. transcribed 전체 텍스트 전달
            4. 새로운 프로세스 생성
        """
        while self.running:
            full = self.recorder.text()
            if not full:
                continue

            self.cancel_event.set()

            # LLM 응답 및 TTS 생성 종료
            if self.llm_thread and self.llm_thread.is_alive():
                self.llm_thread.join()

            self.cancel_event = threading.Event()
            asyncio.run_coroutine_threadsafe(
                self.ws.send_text(json.dumps({'type': 'fullSentence', 'text': full})),
                self.loop
            )

            self.llm_thread = threading.Thread(
                target=self._llm_tts_worker,
                args=(full, self.cancel_event),
                daemon=True
            )
            self.llm_thread.start()

    def _llm_tts_worker(self, full: str, cancel_event: threading.Event):
        """
            (async) LLM 응답 생성, TTS 음성 생성 및 전달
        """
        tts_index = 0

        # TTS 생성 및 전송
        def send_tts(sentence: str):
            nonlocal tts_index
            if cancel_event.is_set():
                return

            text = emoji.replace_emoji(sentence, replace='')
            text = re.sub(r'(__|\*\*|\*|`|~~|!\[.*?\]\(.*?\)|\[.*?\]\(.*?\))', '', text)
            text = text.replace('\n', '').replace('\r', '').replace('\t', '')
            text = re.sub(r'(?<=\d)[.,?!:]', '', text)
            text = re.sub(r'[^\uAC00-\uD7A3\u3131-\u318F0-9A-Za-z,.!? ]+', '', text)
            clean = re.sub(r'\s+', ' ', text).strip()
            if not clean:
                return

            # GPT-SoVITS TTS 생성
            gsv = importlib.import_module("gpt_sovits_api")
            pcm = bytearray()
            for chunk in gsv.get_tts_wav(
                ref_wav_path="/home/keem/sample.wav",
                prompt_text="なるべく、教師との無駄なやり取りを発生させたくないですもんね。",
                prompt_language="ja",
                text=clean,
                text_language="ko"
            ):
                if cancel_event.is_set():
                    return
                pcm.extend(chunk)

            # 클라이언트 전송
            payload = {
                "type": "audio_chunk",
                "index": tts_index,
                "audio_base64": base64.b64encode(pcm).decode('ascii')
            }
            asyncio.run_coroutine_threadsafe(
                self.ws.send_text(json.dumps(payload)),
                self.loop
            )
            tts_index += 1

        # LLM 스트리밍 + TTS 버퍼 처리
        tts_buffer = TTSBuffer(send_tts)
        for chunk in chat_stream(full):
            # 마이크 입력 들어오면 부모에서 cancel_event set 하고 join (종료 대기)
            if cancel_event.is_set():
                return
            # LLM 응답 청크 전송
            asyncio.run_coroutine_threadsafe(
                self.ws.send_text(json.dumps({'type': 'response_chunk', 'text': chunk})),
                self.loop
            )
            tts_buffer.feed(chunk)

        # 완료 신호 및 남은 버퍼 flush
        if not cancel_event.is_set():
            asyncio.run_coroutine_threadsafe(
                self.ws.send_text(json.dumps({'type': 'response_done'})),
                self.loop
            )
            tts_buffer.flush()

    def stop(self):
        # 세션 종료
        self.running = False
        self.cancel_event.set()
        if self.llm_thread and self.llm_thread.is_alive():
            self.llm_thread.join()

    # 현재 진행중인 프로세스 취소 & 클라이언트에 신호 전송
    def halt_current_process(self):
        if not self.running :
            return
        self.cancel_event.set()
        if self.llm_thread and self.llm_thread.is_alive():
            self.llm_thread.join()
        # 클라이언트에 종료 메시지 전달
        asyncio.run_coroutine_threadsafe(
            self.ws.send_text(json.dumps({'type': 'cancel_audio'})),
            self.loop
        )

    # VAD 감지 - 기존 생성 프로세스 중지 -> 새로운 생성 프로세스 시작
    def on_vad_start(self):
        self.halt_current_process()
        return
