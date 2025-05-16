from fastapi import Request, FastAPI

from app.api.common_response import CommonResponse
import logging

from app.exception.exceptions import ControlledException

app = FastAPI()

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exception: Exception):
    logging.error(f"[{request.method}] {request.url} 에서 에러 발생: {exception}")
    return CommonResponse(
        code=-500,
        message="알 수 없는 에러",
    )

@app.exception_handler(ControlledException)
async def controlled_exception_handler(request: Request, exception: ControlledException):
    logging.error(f"[{request.method}] {request.url} 에서 에러 발생: {exception}")
    return CommonResponse(
        code= exception.error_code.code,
        message= exception.error_code.message,
    )