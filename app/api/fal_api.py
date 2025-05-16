from fastapi import APIRouter, HTTPException
from app.services.fal_handler import generate_image, generate_image_prompt

router = APIRouter(prefix="/", tags=["image"])

@router.post("/image")
async def create_image_prompt(ko_prompt: str):
    try:
        prompt = generate_image_prompt(ko_prompt)
        return generate_image(prompt=prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
