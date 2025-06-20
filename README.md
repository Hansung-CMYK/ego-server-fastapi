# Speak-To-You
- 실제 사람과 대화가 부담스러울 때, AI 페르소나 'EGO'를 통해 편하게 소통하는 소셜 서비스
- Key Feature

# Settings
- 음성 보이스 설치?
- ???

# How to Install
- ???

# API Documentation
- Swagger UI : http://localhost:8000/docs

# Directory Structure
```
.
├── app
│   ├── __init__.py			    # 패키지 초기화
│   ├── main.py			        # FastAPI 앱 진입점 (uvicorn main:app 실행)
│   ├── dependencies.py		    # 공통 의존성 (DB, 인증, 공통 모듈 DI)
│   ├── routers			        # 비즈니스 도메인 별 API 엔드포인트
│   │   ├── __init__.py
│   │   └── items			    # Items 도메인
│   │	    ├── items.py		            # APIRouter 등록
│   │	    ├── items_controller.py	        # HTTP 요청 처리 (입출력)
│   │	    ├── items_repository.py	        # DB 처리 계층 (ORM)
│   │       ├── items_service.py	        # 비즈니스 로직
│   │	    └── dto			    # 요청/응답 모델
│   │		    ├── items_request.py
│   │		    └── items_response.py
│   └── internal				# 내부용 전용 API/모듈
│		├── __init__.py
│		├── exception			# 예외 처리 시스템
│		│	├── exception_handler.py	# FastAPI 글로벌 예외 처리 등록
│		│	└── error_code.py	# 에러코드 정의 및 매핑
│		├── logger.py			# 로깅 설정 (Logger 커스터마이징)
│		└── admin			    # Admin 도메인
│			├── admin_controller.py
│			├── admin_repository.py
│	   		├── admin_service.py
│			└── dto
│				├── admin_request.py
│				└── admin_response.py
├── config                      # 시스템 환경 설정 및 외부 라이브러리 초기화
│	├── database
│	│	├── database1.py		# 첫 번째 DB 클라이언트 (예: PostgreSQL, MySQL 등)
│	│	└── database2.py		# 추가 DB 클라이언트 (예: Redis, MongoDB 등)
│	└── llm
│		└── llm1.py			    # LLM 모델 초기화 및 클라이언트 관리
├── Dockerfile					# 도커 이미지 생성을 위한 DockerFile
└── test					    # 테스트 코드 (pytest 기반 유닛테스트/통합테스트)
```

# Git Strategy
### Branch Strategy
| Name    | Description      |
|---------|------------------|
| main    | release version  |
| develop | develop version  |
| feat    | feature version  |

### PR Strategy
PR 전략에 대한 자세한 정보는 [PR Strategy 바로가기](docs/pr_strategy.md)를 참고해주세요.

### Issue Strategy
Issue 전략에 대한 자세한 정보는 [Issue Strategy 바로가기](docs/issue)strategy.md)를 참고해주세요.

# Dependency & Library
| Name  | Description                                           | Version |
|-------|-------------------------------------------------------|---------|
| name  | description                                           | version |
| isort | Import sorting tool for Python, compliant with PEP 8. | 6.0.1   |

# Support
### E-Mail
- team_email

### Contributer
| Name | Role                          | Description   | Link                                         |
|------|-------------------------------|---------------|----------------------------------------------|
| 이준희  | Team Leader, PM, BE, etc...   | 안녕하세요 이준희입니다. | [HS-JNYLee](https://github.com/HS-JNYLee)    |
| 김명준  | AI, DB, Cluster Administrator | 안녕하세요 김명준입니다. | [KimMyeongjun](https://github.com/gomj-repo) |
| 김상준  | AI, CI/CD, Kafka manager      | 안녕하세요 김상준입니다. | [Keem](https://github.com/6-keem)            |
| 김재호  | FE, BE, DB, TC                | 안녕하세요 김재호입니다. | [KJH0506](https://github.com/KJH0506)        |

# License
| Name | License | CopyRight  |
|------|---------|------------|
| name | license | copy_right |