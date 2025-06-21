from json import JSONDecodeError

from requests import Response

from app.internal.logger.logger import logger


def parse_json(response:Response, title:str):
    """
    요약:
        API로 반환된 JSON을 객체로 파싱하고, 에러를 분석하는 함수

    Parameters:
        response(Response): API의 반환 결과
        title(str): 에러 발생 시 나타날 문구
    """
    try:
        result = response.json()
        if result["code"] != 200:
            # LOG. 시연용 로그
            logger.exception(msg=f"\n\nExternal API Error [{title}]\n{result["message"]}\n")
        return result["data"]
    except JSONDecodeError:
        # LOG. 시연용 로그
        logger.exception(msg=f"\n\nExternal API Error [{title}]\n{response}\n")
    except KeyError:
        # LOG. 시연용 로그
        logger.exception(msg=f"\n\nExternal API Error [{title}]\n{response}\n")