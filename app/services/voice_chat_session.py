import os, sys, uuid, json, threading, importlib, asyncio, re, base64, emoji

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

get_shared_whisper_model

class VoiceChatSession:
    def __init__(self, websocket):
        self.id = str(uuid.uuid4())
        self.ws = websocket
        self.loop = asyncio.get_running_loop()
        self.recorder = AudioToTextRecorder(**self._recorder_config(),
                                            on_realtime_transcription_stabilized=self.on_text_detected)
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
        while self.running:
            full = self.recorder.text()
            if not full:
                continue

            # STT 최종 문장
            asyncio.run_coroutine_threadsafe(
                self.ws.send_text(json.dumps({'type': 'fullSentence', 'text': full})),
                self.loop
            )

            # TTS 전송 함수 정의
            tts_chunk_index = 0
            def send_tts_to_ws(sentence: str):
                nonlocal tts_chunk_index
                clean = emoji.replace_emoji(sentence)
                clean = re.sub(r'(__|\*\*|\*|`|~~|!\[.*?\]\(.*?\)|\[.*?\]\(.*?\))', '', clean)
                clean = re.sub(r'\s+', ' ', clean).strip().replace(':',',')
                if not clean: return

                gsv = importlib.import_module("gpt_sovits_api")
                pcm = bytearray()
                for chunk in gsv.get_tts_wav(
                    ref_wav_path="/home/keem/sample.wav",
                    prompt_text="なるべく、教師との無駄なやり取りを発生させたくないですもんね。",
                    prompt_language="ja",
                    text=clean,
                    text_language="ko"
                ):
                    pcm.extend(chunk)
                payload = {
                    "type": "audio_chunk",
                    "index": tts_chunk_index,
                    "audio_base64": base64.b64encode(pcm).decode('ascii')
                }
                asyncio.run_coroutine_threadsafe(self.ws.send_text(json.dumps(payload)), self.loop)
                tts_chunk_index += 1

            tts_buffer = TTSBuffer(send_tts_to_ws)

            # LLM 응답 & TTS 버퍼링
            for chunk in chat_stream(full):
                asyncio.run_coroutine_threadsafe(
                    self.ws.send_text(json.dumps({'type': 'response_chunk', 'text': chunk})),
                    self.loop
                )
                tts_buffer.feed(chunk)

            # 응답 완료
            asyncio.run_coroutine_threadsafe(
                self.ws.send_text(json.dumps({'type': 'response_done'})), self.loop
            )
            tts_buffer.flush()

    def stop(self):
        self.running = False
