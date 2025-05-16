from fastapi import Request, FastAPI
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse
from app.api.common_response import CommonResponse
import logging

from app.exception.exceptions import ControlledException

def register_exception_handlers(app: FastAPI):
    @app.exception_handler(ControlledException)
    async def controlled_exception_handler(request: Request, exception: ControlledException):
        logging.error(f"[{request.method}] {request.url} 에서 에러 발생: {exception}")
        body = CommonResponse(
            code= exception.error_code.code,
            message= exception.error_code.message,
        )
        return JSONResponse(
            status_code=400,
            content=jsonable_encoder(body)
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exception: Exception):
        logging.error(f"[{request.method}] {request.url} 에서 에러 발생: {exception}")
        body = CommonResponse(
            code=-500,
            message="알 수 없는 에러",
        )
        return JSONResponse(
            status_code=400,
            content=jsonable_encoder(body)
        )