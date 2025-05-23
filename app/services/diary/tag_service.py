from __future__ import annotations

import os

import json
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

import requests
from app.models.default_model import kiwi
from typing import List

from app.models.txtnorm.embedding_model import embedding_model
from app.services.diary.tag_embedding_service import load_index

load_dotenv()
SPRING_URI = os.getenv('SPRING_URI')

__KEYS, __EMBEDS = load_index()

def sentence_embedding(stories: list[str]) -> np.ndarray:
    """
    요약:
        사용자 메세지를 정제하는 함수이다.
    
    Parameters:
        stories(list[str]): 사용자의 원본 문장

    Returns:
        사용자 문장의 명사만 추출해서 임베딩한 결과
    """
    # 명사만 추출해 노이즈 감소
    user_chat_logs = "\n".join(stories)
    nouns = [tok.form for sent in kiwi.analyze(user_chat_logs) for tok in sent[0] if tok.tag.startswith("NN")]
    message = " ".join(nouns) if nouns else user_chat_logs
    return embedding_model.embeded_documents([message])[0].astype(np.float32)

def search_tags(
    embedded_user_chat_logs: np.ndarray,
    top_k: int = 5,
    min_sim: float = 0.4,
) -> List[str]:
    """
    요약:
        벡터 임베딩된 문장과 가장 가까운 유사도의 태그들을 조회하는 함수

    Parameters:
        embedded_user_chat_logs(np.ndarray[float32]): (dim,) 문장 임베딩
        top_k(int): 반환할 최대 키워드 개수
        min_sim(float): 유사도 필터(코사인 기준)
    """
    # NOTE 1. torch Tensor 변환
    msg_vec  = torch.from_numpy(embedded_user_chat_logs).unsqueeze(0) # (1, dim)
    key_vecs = torch.from_numpy(__EMBEDS) # (k, dim)

    # NOTE 2. L2 정규화 후 코사인 유사도
    msg_vec  = F.normalize(msg_vec,  dim=1)
    key_vecs = F.normalize(key_vecs, dim=1)
    sims = torch.mm(msg_vec, key_vecs.T).squeeze(0) # (k,)

    # NOTE 3. Top-k 추출 + threshold 필터
    top_scores, idx = torch.topk(sims, k=min(top_k, sims.size(0)))
    return [
        __KEYS[i]
        for j, i in enumerate(idx) # enumerate로 (순번, index) 튜플 획득
        if top_scores[j] >= min_sim # 임계값 이상만
    ]

def patch_tags(ego_id:int, tags:list[str]):
    """
    요약:
        사용자의 태그 정보를 업데이트 하기 위한 함수이다.

    Parameters:
        ego_id(int): 태그를 추가할 에고의 아이디
        tags(list[str]): 추가할 태그 목록
    """
    url = f"{SPRING_URI}/api/v1/ego"
    update_data = {"id": ego_id, "personalityList": tags}
    headers = {"Content-Type": "application/json"}
    return requests.patch(url=url, data=json.dumps(update_data), headers=headers)
