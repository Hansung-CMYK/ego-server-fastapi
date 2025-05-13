from datetime import datetime

from app.models.postgres_client import postgres_client

class PersonaStore:
    """
    세션(또는 사용자) 단위 페르소나 메모리.

    주요 특징
    - `$set` / `$unset` 두 연산자만 지원해 변경, 삭제를 명시적으로 구분
    - 리스트 항목 제거, 딕셔너리 키 제거 재귀 지원
    """

    """
    store은 다음과 같은 형식으로 저장된다.
    {
        (<persona_id>, <persona_name>, <persona_data>)
    }
    """
    store: dict[int, list] = {}

    # 키값으로 사용될 수 있는 정보들 (화이트리스트)
    # PersonalLlmModel의 허용 최상위 키와는 별개이다.
    __ALLOWED_TOP_KEYS = {
        "name", "age", "gender",
        "location", "likes", "dislikes",
        "mbti", "personality", "goal",
        "updated_at", "$set", "$unset"
    }

    # 프롬프트에 사용할 기본 페르소나 정보
    __DEFAULT_PERSONA = {
        "name": None,
        "age": None,
        "gender": None,
        "location": None,
        "likes": [],
        "dislikes": [],
        "mbti": None,
        "personality": [],
        "goal": [],
        "updated_at": None,
    }

    def get_persona(self, persona_id:int) -> dict:
        """
        사용자의 페르소나 정보를 반환하는 함수
        """
        if self.store.get(persona_id) is None: # 기존 store에 persona_id가 저장되어 있지않다면,
            new_persona = postgres_client.select_persona_to_id(persona_id) # postgres에서 정보를 가져와서,
            self.store[persona_id] = new_persona # store에 저장한다.

        return self.store[persona_id][2]

    def update(self, persona_id:int, delta_persona: dict):
        """
        delta(dict)를 받아 페르소나 in‑place 갱신
        """
        # dict 키값에 지정된 키(_ALLOWED_TOP_KEYS) 값만 남겨둔다.
        delta_dict = {key: value for key, value in delta_persona.items() if key in self.__ALLOWED_TOP_KEYS}

        # $unset 먼저 처리
        if "$unset" in delta_dict:
            self.__apply_unset(
                original_data=self.store[persona_id][2],
                unset_data=delta_dict.pop("$unset")
            )

        # 새로운 페르소나 데이터 추가.
        if "$set" in delta_dict:
            self.__apply_set(
                original_data=self.store[persona_id][2],
                set_data=delta_dict.pop("$set")
            )

        # 업데이트 된 시간 변경
        self.store[persona_id][2]["updated_at"] = datetime.now().isoformat()

        # 데이터베이스에 저장
        postgres_client.update_persona(persona_id=persona_id, persona_json=self.store[persona_id][2])

    @staticmethod
    def __apply_unset(original_data: dict, unset_data: dict) -> None:
        """Dictionary 안에서 값을 **제거**하는 함수.

        * 딕셔너리   → 재귀로 내려가 서브키 제거
        * 리스트     → 주어진 항목(targets)을 제거, 비면 키 통째 삭제
        * 원시 타입  → 키 통째 삭제
        """
        print("-----unset_data-----")
        print(unset_data)

        for key, value in unset_data.items():
            # 오리지널 데이터에 존재하지 않는 값은 제외한다.
            if key not in original_data: continue

            # 키 값의 내부 데이터가 list인 경우
            if isinstance(original_data[key], list):
                targets = value if isinstance(value, list) else [value]
                original_data[key][:] = [x for x in original_data[key] if x not in targets]
            else:
                original_data[key] = None

    @staticmethod
    def __apply_set(original_data: dict, set_data: dict) -> None:
        """
        Dictionary 안에 있는 값을 머지하는 함수이다.

        :param original_data: 기존 페르소나 데이터
        :param set_data: 추가될 페르소나 데이터
        """
        print("-----set_data-----")
        print(set_data)

        for key, value in set_data.items():
            # 키 값의 내부 데이터가 list인 경우
            if key in original_data and isinstance(original_data[key], list) and isinstance(value, list):
                original_data[key].extend(x for x in value if x not in original_data[key])
            else:
                original_data[key] = value # 중복되는 키 값을 가진 데이터는 최신 데이터로 교체

persona_store = PersonaStore()