import gc
import logging
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

logger = logging.getLogger(__name__)

_STT_SEMAPHORE = threading.Semaphore(value=1)
_recorder_instance: AudioToTextRecorder | None = None
_full_thread: threading.Thread | None = None

_subscribers_rt: list[callable] = []
_subscribers_full: list[callable] = []

def _dispatch_realtime(text: str):
    for cb in list(_subscribers_rt):
        try:
            cb(text)
        except Exception:
            pass

def _dispatch_full_loop():
    global _recorder_instance
    while _recorder_instance is not None:
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
) -> AudioToTextRecorder | None:
    global _recorder_instance, _full_thread

    if not _STT_SEMAPHORE.acquire(blocking=False):
        logger.warning("STT 엔진이 사용 중입니다")
        return None

    # 최초 생성 시
    if _recorder_instance is None:
        cfg = cfg.copy()
        cfg['on_realtime_transcription_stabilized'] = _dispatch_realtime
        _recorder_instance = AudioToTextRecorder(**cfg)

        _full_thread = threading.Thread(target=_dispatch_full_loop, daemon=True)
        _full_thread.start()

    else:
        try:
            # 내부 상태 초기화
            _recorder_instance.reset()  # ❗ 반드시 AudioToTextRecorder에 이 메서드 구현 필요
        except AttributeError:
            logger.warning("AudioToTextRecorder에 reset() 메서드가 없습니다")

    # 콜백 초기화 및 재등록
    _subscribers_rt.clear()
    _subscribers_full.clear()
    _subscribers_rt.append(on_realtime)
    _subscribers_full.append(on_full_sentence)

    return _recorder_instance

def release_stt_recorder():
    global _recorder_instance, _subscribers_rt, _subscribers_full

    if _recorder_instance is not None:
        try:
            _recorder_instance.stop()
            _recorder_instance.join()
        except Exception:
            pass

        try:
            _recorder_instance.flush()  # 내부 버퍼 정리용, 없다면 무시
        except Exception:
            pass

    _subscribers_rt.clear()
    _subscribers_full.clear()

    try:
        _STT_SEMAPHORE.release()
    except ValueError:
        pass

    gc.collect()
