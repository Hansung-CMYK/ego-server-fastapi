import multiprocessing as mp

mp.set_start_method('spawn', force=True)
import subprocess
import os

def get_free_gpu():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        memory_free = [int(x) for x in result.strip().split("\n")]
        best_gpu = max(range(len(memory_free)), key=lambda i: memory_free[i])
        return str(best_gpu)
    except Exception as e:
        print("GPU 자동 선택 실패:", e)
        return "0" 

os.environ["CUDA_VISIBLE_DEVICES"] = get_free_gpu()

from app.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port='8000')

