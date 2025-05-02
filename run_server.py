import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from app.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

