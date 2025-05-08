import logging

import stanza

class NerModel:
    """
    한국어 Stanza UD 분석을 이용해 [주어, 목적어(또는 보어), 전체 문장] 리스트를 추출하는 스크립트.

    주요 기능
    ----------
    1. **Stanza 파이프라인 초기화** : `tokenize`, `pos`, `depparse`, `lemma` 모듈을 로드해 형태소와 의존구문 정보를 수집합니다.
    2. **명사구(phrase) 확장**       : 주·목적어 핵심 토큰에 붙는 수식어(복합명사, 관형어 등)를 재귀적으로 포함해 자연스러운 명사구를 만듭니다.
    3. **주어·목적어(또는 보어) 선택** : 여러 의존 라벨 패턴을 우선순위 규칙으로 분석해 빠진 태그까지 최대한 보완합니다.
    4. **삼중항 리스트 반환**       : `[주어, 목적어/보어, 원문]` 형태로 각 문장을 정리해 후처리(RAG, DB 삽입 등)에 바로 쓸 수 있습니다.

    ※ 목적어와 보어가 한 문장에 동시에 있으면 "목적어"를 우선 사용합니다.
    ※ 정확도가 중요한 경우 `_PARTICLES`, `SUBJ_TAGS`, `OBJ_TAGS` 를 과제/도메인에 맞춰 확장하세요.
    """
    # 2. 상수 설정
    # 명사구 확장 시 포함할 의존 태그 집합
    __PHRASE_DEPS = {"compound", "nmod", "amod", "advmod", "obl", "conj"}

    # 주어 추정을 위한 대표 주격 조사
    __SUBJ_JOSA = ("은", "는", "이", "가")

    def __init__(self):
        # ner 호출을 위한 지연시간 서비스 실행시 (1~2) 지연
        self.__nlp = stanza.Pipeline(
            lang="ko",
            processors="tokenize,pos,depparse,lemma",  # ㆍ'lemma'는 depparse의 의존성이므로 processors 순서에 포함
            batch_size=32,  # 대량 문장 처리 시 속도에 영향 (문장 길이*batch_size 메모리)
            verbose=False
        )
        self.__nlp("")

        # ㆍStanza 내부 디버그 로그(INFO) 노이즈를 막기 위해 WARNING 이상만 출력
        logging.getLogger("stanza").setLevel(logging.WARNING)

    def __expand_phrase(self, tok, sent):
        """
            주어진 토큰(tok)을 중심으로 수식·병렬 의존어를 재귀 수집하여 자연스러운 명사구 문자열을 반환.
            같은 토큰 중복을 막기 위해 `seen` 집합 사용
            `parts` 리스트에 (id, text)를 모아 토큰 id 기준 오름차순 정렬 후 join
        """
        stack = [tok]  # DFS용 스택 초기화 (현재 토큰부터)
        parts = []  # 최종 명사구로 만들 토큰들 저장
        seen = set()  # 중복 방지

        while stack:
            node = stack.pop()  # 스택에서 하나 꺼냄
            if node.id in seen:
                continue
            seen.add(node.id)  # 중복 방지용
            parts.append((node.id, node.text))  # 현재 단어 기록

            for ch in sent.words:  # 문장의 다른 단어들 중
                if ch.head == node.id and ch.deprel in self.__PHRASE_DEPS:
                    stack.append(ch)  # 자식 노드면서 수식 역할이면 스택에 넣음

        return " ".join(t for _, t in sorted(parts))  # 토큰 id 순으로 정렬 후 문장 생성

    def __select_subject(self, sent):
        """
            주어(명사구)를 추정하는 함수
            반환값이 빈문자열("") 인 경우 주어 없음 의미

            우선순위: nsubj(주어 명사) > dislocated(이탈 주어/목적어)/topic(화제 주어) > csubj(절 주어) > 기타 백업 규칙
        """
        # 표준 주어 태그(nsubj)
        for deps in ("nsubj", "dislocated", "topic", "csubj"):
            for w in sent.words:
                if w.deprel in deps:
                    return self.__expand_phrase(w, sent)

        # 문장 첫 토큰이 대명사/고유명 + 조사 없음 → 화자/호명
        first = sent.words[0]
        if first.upos in {"PRON", "PROPN"} and not first.text.endswith(self.__SUBJ_JOSA):
            return self.__expand_phrase(first, sent)

        # 오태깅 보정: compound(복합 명사 수식어)/acl(형용사절) 이면서 주격 조사 포함
        for w in sent.words:
            if w.deprel in {"compound", "acl"} and w.text.endswith(self.__SUBJ_JOSA):
                return self.__expand_phrase(w, sent)

        # 임시: 장소 부사어(…에서)를 주어 대용 (필요시 제외)
        # 만약 넣을만한 문장이 없을 땐, 대체 용어를 추가한다.
        for w in sent.words:
            if w.deprel == "obl" and w.text.endswith("에서"):
                return self.__expand_phrase(w, sent)

        # 모든 규칙에서 반환되지 않았음은 문장에 주어가 없음을 의미한다.
        return ""

    def __select_object(self, sent):
        """
            목적어 구를 추정하는 함수
            반환값이 빈문자열("") 인 경우 목적어 없음 의미

            우선순위: obj(기본 목적격) > topic(은/는)
        """
        # 1) 기본 목적격
        for w in sent.words:
            if w.deprel == "obj":
                return self.__expand_phrase(w, sent)
        # 2) 대조·주제 topic을 목적어처럼 사용.
        # 만약 주어에 다른 명사가 포함되었다면, 여기에 저장
        for w in sent.words:
            if w.deprel == "topic" and w.upos in {"NOUN", "PROPN"}:
                return self.__expand_phrase(w, sent)

        return ""

    def __select_complement(self, sent):
        """
            보어를 추정하는 함수
            반환값이 빈문자열("")인 경우 보어 없음 의미

            우선순위: cop(연결 동사{이다, 되다}) > csubj(절 문장{~는 것, ~기}) > 기타 백업 규칙
        """
        # 1) Copula("이다", "되다" 등) 패턴
        for w in sent.words:
            if w.deprel == "cop":
                head = sent.words[w.head - 1]  # 연결 동사 앞 단어가 보어
                return self.__expand_phrase(head, sent)
        # 2) 남은 csubj
        for w in sent.words:
            if w.deprel == "csubj":
                return self.__expand_phrase(w, sent)

        # 임시: 장소 부사어(…에서)를 주어 대용 (필요시 제외)
        # 만약 넣을만한 문장이 없을 땐, 대체 용어를 추가한다.
        for w in sent.words:
            if w.deprel == "obl" and w.text.endswith("에서"):
                return self.__expand_phrase(w, sent)

        return ""

    def extract_triplets(self, text: str):
        """
        텍스트(여러 문장 가능)를 받아 [[주어, 목적어/보어, 원문],...] 목록 반환.

        Arguments:
            text: 반환받고 싶은 문장
        """
        doc = self.__nlp(text.strip())  # stanza로 문장 분석
        triplets = []
        sentences = []
        for sent in doc.sentences:
            subj = self.__select_subject(sent)  # 주어
            obj = self.__select_object(sent)  # 목적어
            comp = "" if obj else self.__select_complement(sent)  # 목적어 없으면 보어 사용
            triplets.append([subj, obj or comp, sent.text])  # obj 우선 저장
            sentences.append(sent.text)
        return {
            "triplets": triplets,
            "relations": sentences
        }