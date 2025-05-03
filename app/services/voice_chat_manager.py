import os
import sys
import asyncio
import threading
import json
import uuid
import re
import emoji
from app.services.tts_infer import get_tts_wav
from app.services.ollama_service import chat_stream

REALTIME_STT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../modules/RealtimeSTT")
)
if REALTIME_STT_PATH not in sys.path:
    sys.path.insert(0, REALTIME_STT_PATH)

import faster_whisper
from faster_whisper import WhisperModel as _OriginalWhisperModel

__whisper_model_cache = {}

def _shared_whisper_model(
    model_size_or_path, device="cuda", compute_type="default",
    device_index=None, download_root=None
):
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

_SENTENCE_END_RE = re.compile(r'^(.*?[\.\!?]+)(.*)$', re.DOTALL)

class TTSBuffer:
    def __init__(self, send_tts_fn):
        self.buffer = ""
        self.send_tts = send_tts_fn 

    def feed(self, chunk: str):
        # 청크 추가
        self.buffer += chunk
        while True:
            match = _SENTENCE_END_RE.match(self.buffer)
            if not match:
                break
            sentence, remainder = match.group(1), match.group(2)
            # 이후에도 문장 부호 인지 검사 아니면 탈출
            if remainder and re.match(r'^[\.!?]', remainder):
                break
            # 전처리 후 TTS 생성
            self.send_tts(sentence)
            self.buffer = remainder

    def flush(self):
        if self.buffer.strip():
            self.send_tts(self.buffer)
        self.buffer = ""

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

            # STT 완료 문장 전송
            asyncio.run_coroutine_threadsafe(
                self.ws.send_text(json.dumps({'type': 'fullSentence', 'text': full})),
                self.loop
            )

            # 이모지 제거, 마크다운 문법 제거, TTS 생성
            def send_tts_to_ws(sentence: str):
                clean = emoji.demojize(str, language='ko')
                clean = re.sub(r'(__|\*\*|\*|`|~~|!\[.*?\]\(.*?\)|\[.*?\]\(.*?\))', '', clean)
                pcm = bytearray()
                for audio_chunk in get_tts_wav(
                    ref_wav_path, clean, prompt_language, clean, text_language
                ):
                    pcm.extend(audio_chunk)
                asyncio.run_coroutine_threadsafe(
                    self.ws.send_bytes(pcm),
                    self.loop
                )

            tts_buffer = TTSBuffer(send_tts_to_ws)

            # LLM 응답 청크 전송
            for chunk in chat_stream(full):
                # 전송
                asyncio.run_coroutine_threadsafe(
                    self.ws.send_text(json.dumps({'type': 'response_chunk', 'text': chunk})),
                    self.loop
                )
                # 버퍼 채우기
                tts_buffer.feed(chunk)

            # 응답 완료 알림
            asyncio.run_coroutine_threadsafe(
                self.ws.send_text(json.dumps({'type': 'response_done'})),
                self.loop
            )

            # 버퍼 비우기
            tts_buffer.flush()

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
