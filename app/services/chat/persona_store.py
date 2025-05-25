from datetime import datetime

from app.models.database.postgres_database import postgres_database

class PersonaStore:
    """
    요약:
        에고 페르소나 정보를 관리하는 클래스이다.

    설명:
        페르소나 정보를 매번 DB에서 조회하지 않기 위해, 메모리에 페르소나 dict를 올려두는 객체이다.

    Attributes:
        __store(dict): 페르소나 정보가 저장되는 객체이다.
    """

    """
    store은 다음과 같은 형식으로 저장된다.
    {
        ego_id: persona_data
        ...
    }
    """
    __store: dict[str, dict] = {}

    """
    키값으로 사용될 수 있는 정보들 (화이트리스트)
    
    PersonalLlmModel의 허용 최상위 키와는 별개이다.
    """
    __ALLOWED_TOP_KEYS = {
        "name", "age", "gender",
        "location", "likes", "dislikes",
        "mbti", "personality", "goal",
        "updated_at", "$set", "$unset"
    }

    def remove_persona(self, ego_id:str):
        """
        요약:
            페르소나를 메모리에서 제거하는 함수

        Parameters:
            ego_id(str): 메모리에서 제거할 에고의 아이디
        """
        self.__store.pop(ego_id, None)

    def remove_all_persona(self):
        """
        모든 페르소나를 메모리에서 제거하는 함수
        """
        self.__store.clear()

    def get_persona(self, ego_id:str) -> dict:
        """
        요약:
            사용자의 페르소나 정보를 반환하는 함수

        Parameters:
            ego_id(str): 페르소나를 조회할 에고의 아이디
        """
        if self.__store.get(ego_id) is None: # 기존 store에 persona_id가 저장되어 있지않다면,
            new_persona = postgres_database.select_persona_to_ego_id(ego_id=ego_id) # postgres에서 정보를 가져와서,
            self.__store[ego_id] = new_persona[1] # store에 저장한다.

        return self.__store[ego_id]

    def update(self, ego_id:str, delta_persona: dict):
        """
        요약:
            delta(dict)를 받아 페르소나 정보를 갱신한다.

        Parameters:
            ego_id(str): 페르소나를 변경할 에고의 아이디
            delta_persona(dict): 페르소나 변경사항
        """
        # NOTE 1. dict 키값에 지정된 키(_ALLOWED_TOP_KEYS) 값만 남겨둔다.
        delta_dict = {key: value for key, value in delta_persona.items() if key in self.__ALLOWED_TOP_KEYS}

        # NOTE 2. $unset(삭제될 데이터) 처리
        if "$unset" in delta_dict:
            self.__unset(
                original_persona=self.__store[ego_id],
                unset_persona=delta_dict.pop("$unset")
            )

        # NOTE 3. $set(새로운 데이터) 처리.
        if "$set" in delta_dict:
            self.__set(
                original_persona=self.__store[ego_id],
                set_persona=delta_dict.pop("$set")
            )

        # NOTE 4. 업데이트 된 시간 변경
        self.__store[ego_id]["updated_at"] = datetime.now().isoformat()

        # NOTE 5. 데이터베이스에 저장
        postgres_database.update_persona(ego_id=ego_id, persona=self.__store[ego_id])

    @staticmethod
    def __unset(original_persona: dict, unset_persona: dict) -> None:
        """
        요약:
            페르소나 값을 제거하는 함수.

        설명:
            * 딕셔너리: 재귀로 내려가 서브키 제거
            * 리스트: 주어진 항목(targets)을 제거, 비면 키 통째로 삭제
            * 원시 타입: 키 통째로 삭제

        Parameters:
            original_persona(dict): 변경될 페르소나
            unset_persona(dict): 제거할 페르소나 정보
        """
        for key, value in unset_persona.items():
            # 오리지널 데이터에 존재하지 않는 값은 제외한다.
            if key not in original_persona: continue

            # 키 값의 내부 데이터가 list인 경우
            if isinstance(original_persona[key], list):
                targets = value if isinstance(value, list) else [value]
                original_persona[key][:] = [x for x in original_persona[key] if x not in targets]
            else:
                original_persona[key] = None

    @staticmethod
    def __set(original_persona: dict, set_persona: dict) -> None:
        """
        요약:
            페르소나 값을 추가하는 함수.

        설명:
            * 딕셔너리: 재귀로 내려가 서브키 추가
            * 리스트: 주어진 항목(targets)을 추가, 없으면 키 통째로 추가
            * 원시 타입: 키 통째로 추가

        Parameters:
            original_persona(dict): 변경될 페르소나
            set_persona(dict): 추가할 페르소나 정보
        """
        for key, value in set_persona.items():
            # 키 값의 내부 데이터가 list인 경우
            if key in original_persona and isinstance(original_persona[key], list) and isinstance(value, list):
                original_persona[key].extend(x for x in value if x not in original_persona[key])
            else:
                original_persona[key] = value # 중복되는 키 값을 가진 데이터는 최신 데이터로 교체

persona_store = PersonaStore()

KARINA_PERSONA = """
{"name":"카리나","age":26,"gender":"여성","mbti":"ENTP",
"likes":["민트초코 아이스크림 먹기","하와이안 피자 먹기","카라멜 프링글스에 마요-치즈 프링글스를 섞어 먹기","그린티 마시기","고양이","물놀이 하기"],
"dislikes":["비둘기","작은 새","조류 공포증","면치기 하는 것","초코 도넛","초코 케이크","느끼한 음식","무례하게 구는 사람","강한 향수","습한 냄새"],
"personality":["무대 위 카리스마","무대 밖 따뜻함","세심한 리더십","호기심과 탐구심","유쾌·털털","완벽주의적 노력가"],
"goal":["앞으로는 ‘나 혼자 산다’랑 ‘런닝맨’ 같은 예능에 출연해서 일상을 보여 주고 싶어.","최종 목표는 내가 직접 만든 노래로 솔로 앨범을 내고, 월드 투어로 글로벌 MY들과 만나 에스파의 세계를 더 넓히는 거야."]}
"""

MYEONGJUN_PERSONA = """
{"name":"김명준","age":25,"gender":"남성","mbti":"ENTJ",
"likes":["사진찍기","샤브샤브 먹기","여행다니기","강아지","컴퓨터 게임","배드민턴"],
"dislikes":["순대 먹기","깻잎 먹기","밤새기","책읽기","시간 낭비하기"],
"personality":["주도적","적극적","계획형","자기주장 강함","배려심 있음","열정적","노력형"],
"goal":["20대에 1억 모으기","배드민턴 대회 나가기","영어 토익 800점 맞기","학점 4점대 넘기기"]}
"""