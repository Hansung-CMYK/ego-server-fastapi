import os
import sys
import threading
import time

# RealtimeSTT 모듈 경로 추가
REALTIME_STT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../modules/RealtimeSTT")
)
if REALTIME_STT_PATH not in sys.path:
    sys.path.insert(0, REALTIME_STT_PATH)

from RealtimeSTT.audio_recorder import AudioToTextRecorder

# 최대 1개 세션만 허용
_STT_SEMAPHORE = threading.Semaphore(value=1)


import logging
logger = logging.getLogger(__name__)

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
    global _recorder_instance
    # 인스턴스가 None이 되면 루프 종료
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
) -> AudioToTextRecorder:
    """
    · 세마포어를 비차단(acquire(blocking=False)) 방식으로 시도합니다.
      실패하면 즉시 RuntimeError 발생.
    · 최초 호출 시 AudioToTextRecorder 인스턴스 생성 + full-loop 스레드 시작.
    · 이후 호출부터는 콜백만 리스트에 추가.
    """
    global _recorder_instance, _full_thread

    # 1) 동시 세션 제어
    if not _STT_SEMAPHORE.acquire(blocking=False):
        logger.warning("STT engine is currently in use by another session; rejecting new request")
        return None

    # 2) 콜백 구독
    _subscribers_rt.append(on_realtime)
    _subscribers_full.append(on_full_sentence)

    # 3) 최초 인스턴스 및 스레드 기동
    if _recorder_instance is None:
        cfg['on_realtime_transcription_stabilized'] = _dispatch_realtime
        _recorder_instance = AudioToTextRecorder(**cfg)

        _full_thread = threading.Thread(
            target=_dispatch_full_loop,
            daemon=True
        )
        _full_thread.start()

    return _recorder_instance

def release_stt_recorder():
    """
    반드시 세션 종료 시 호출해야 합니다.
    - recorder.stop()/join() 호출
    - 인스턴스·콜백 리스트 초기화
    - 세마포어 해제
    """
    global _recorder_instance, _subscribers_rt, _subscribers_full, _full_thread

    # 1) recorder 정지
    if _recorder_instance is not None:
        try:
            _recorder_instance.stop()
        except Exception:
            pass
        # 만약 join()이 필요하면 호출
        try:
            _recorder_instance.join()
        except Exception:
            pass

    # 2) 인스턴스 제거 → dispatch loop 종료
    _recorder_instance = None

    # 3) 콜백 리스트 초기화
    _subscribers_rt.clear()
    _subscribers_full.clear()

    # 4) 세마포어 해제
    try:
        _STT_SEMAPHORE.release()
    except ValueError:
        pass
