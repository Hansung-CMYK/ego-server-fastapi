import os
import sys

import faster_whisper
from faster_whisper import WhisperModel as _OriginalWhisperModel

__whisper_model_cache = {}

def get_shared_whisper_model(
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

# faster_whisper 내부에서 바로 캐시 모델이 쓰이도록 패치
faster_whisper.WhisperModel = get_shared_whisper_model
