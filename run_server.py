from app.main import app
import uvicorn

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
bert_path = "roberta_local" 

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

