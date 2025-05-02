
import numpy as np
from scipy.signal import resample

def decode_and_resample(data: bytes, sr: int, tr: int) -> bytes:
    arr = np.frombuffer(data, np.int16)
    tgt = int(len(arr) * tr / sr)
    out = resample(arr, tgt)
    return out.astype(np.int16).tobytes()
