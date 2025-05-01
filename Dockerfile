# python 이미지
FROM python:3.9

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 프로젝트 전체 복사
COPY . .

# 컨테이너가 시작될 때 실행할 명령어 (FastAPI 앱 실행)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
