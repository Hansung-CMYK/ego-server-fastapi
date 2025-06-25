from fastapi import APIRouter, HTTPException

from config.keem.image.image_generator import ImageGenerator

router = APIRouter()

@router.post("/image")
async def create_image_prompt(prompt_ko: str):
    try:
        return ImageGenerator.generate(prompt_ko)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
