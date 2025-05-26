import os
import sys
import threading
import time

REALTIME_STT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../modules/RealtimeSTT")
)
if REALTIME_STT_PATH not in sys.path:
    sys.path.insert(0, REALTIME_STT_PATH)

from RealtimeSTT.audio_recorder import AudioToTextRecorder

_recorder_instance: AudioToTextRecorder | None = None
_subscribers_rt: list[callable] = []
_subscribers_full: list[callable] = []
_full_thread: threading.Thread | None = None

def _dispatch_realtime(text: str):
    for cb in list(_subscribers_rt):
        try:
            cb(text)
        except Exception:
            pass

def _dispatch_full_loop():
    while True:
        full = _recorder_instance.text()
        if full:
            for cb in list(_subscribers_full):
                try:
                    cb(full)
                except Exception:
                    pass
        time.sleep(0.01)

def get_stt_recorder(
    cfg: dict,
    on_realtime: callable,
    on_full_sentence: callable
) -> AudioToTextRecorder:
    """
    · 최초 호출: AudioToTextRecorder를 만들고
      - on_realtime_transcription_stabilized는 _dispatch_realtime 으로 설정
      - _dispatch_full_loop 스레드도 띄움
    · 이후 호출: 녹음기 재생성 없이, on_realtime / on_full_sentence 콜백만 구독(list에 추가)
    """
    global _recorder_instance, _full_thread

    if _recorder_instance is None:
        cfg['on_realtime_transcription_stabilized'] = _dispatch_realtime
        _recorder_instance = AudioToTextRecorder(**cfg)

        _full_thread = threading.Thread(
            target=_dispatch_full_loop,
            daemon=True
        )
        _full_thread.start()

    _subscribers_rt.append(on_realtime)
    _subscribers_full.append(on_full_sentence)
    return _recorder_instance
