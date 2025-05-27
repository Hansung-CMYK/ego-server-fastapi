import math
import numpy as np
from scipy.signal import resample, resample_poly
import wave

def decode_and_resample(data: bytes, sr: int, tr: int) -> bytes:
    arr = np.frombuffer(data, np.int16)
    tgt = int(len(arr) * tr / sr)
    out = resample(arr, tgt)
    return out.astype(np.int16).tobytes()

def decode_and_resample_v2(
    pcm_bytes: bytes,
    orig_sr: int,
    target_sr: int
) -> bytes:
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if orig_sr != target_sr:
        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        audio = resample_poly(audio, up, down)

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


def save_wav(
    pcm_bytes: bytes,
    sample_rate: int,
    path: str
):
    """
    Raw PCM int16 little-endian 바이트를 mono WAV 파일로 저장합니다.

    Args:
        pcm_bytes: PCM 데이터 (int16 LE)
        sample_rate: 샘플레이트 (예: 24000)
        path: 저장할 .wav 파일 경로
    """
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)            # mono
        wf.setsampwidth(2)            # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)