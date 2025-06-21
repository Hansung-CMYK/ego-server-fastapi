import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse

from app.internal.exception.error_code import ControlledException
from config.common.common_response import CommonResponse


def register_exception_handlers(app: FastAPI):
    @app.exception_handler(ControlledException)
    async def controlled_exception_handler(request: Request, exception: ControlledException)->JSONResponse:
        """
        요약:
            ControlledException이 raise될 때, 처리되는 예외처리 로직

        설명:
            예외로그를 Console에 출력합니다.
            HTTP Code는 400이며, 내부에 CommonResponse가 함께 전달됩니다.

        Parameters:
            exception(ControlledException): 발생한 예외에 대한 정보를 가진 객체
        """
        # logging.exception은 자동으로 traceback을 포함해 로그를 찍어준다.
        logging.exception(msg=f"\n\n[ControlledException 예외 발생]\n[{request.method}] {request.url} 에서 에러 발생: {exception}\n")

        body = CommonResponse(
            code= exception.error_code.code,
            message= exception.error_code.message,
        )

        return JSONResponse(
            status_code=400,
            content=jsonable_encoder(body)
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exception: Exception)->JSONResponse:
        """
        요약:
            알 수 없는 Exception이 raise될 때, 처리되는 예외처리 로직

        설명:
            예외로그를 Console에 출력합니다.
            HTTP Code는 500이며, 내부에 CommonResponse가 함께 전달됩니다.

        Parameters:
            exception(ControlledException): 발생한 예외에 대한 정보를 가진 객체
        """
        logging.exception(f"\n\n[{request.method}]\n{request.url} 에서 에러 발생: {exception}\n")
        body = CommonResponse(
            code=500,
            message="알 수 없는 에러",
        )
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder(body)
        )