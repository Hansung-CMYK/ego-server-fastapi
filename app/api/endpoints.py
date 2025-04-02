from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
def get_status():
    return {"status": "GPU server is up and running"}
