from fastapi import FastAPI
from app.api import endpoints

app = FastAPI(
    title="에고",
    description="FastAPI Server",
    version="0.0.1"
)

app.include_router(endpoints.router)

# 루트 엔드포인트
@app.get("/")
def read_root():
    return {"message": "200 OK"}
