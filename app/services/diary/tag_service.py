from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from app.models.default_model import kiwi
from typing import List

from app.models.normalization.embedding_model import embedding_model
from app.services.diary.tag_embedding_service import load_index

__KEYS, __EMBEDS = load_index()

def sentence_embedding(user_chat_logs: str) -> np.ndarray:
    """
    사용자 메세지를 정제하는 함수이다.
    
    Parameters:
        user_chat_logs: 사용자의 원본 문장

    Returns:
        사용자 문장의 명사만 추출해서 임베딩한 결과
    """
    # 명사만 추출해 노이즈 감소
    nouns = [tok.form for sent in kiwi.analyze(user_chat_logs) for tok in sent[0] if tok.tag.startswith("NN")]
    message = " ".join(nouns) if nouns else user_chat_logs
    return embedding_model.embeded_documents([message])[0].astype(np.float32)


def rank_tags(
    embedded_user_chat_logs: np.ndarray,
    top_k: int = 5,
    min_sim: float = 0.30,
) -> List[str]:
    """
    Parameters:
        embedded_user_chat_logs: 사용자 문장의 명사만 추출해서 임베딩한 결과
        top_k: 반환 개수 (default: 5)
        min_sim: 유사도 필터 기준(default: 0.30)

    Returns:
        List[<keyword>]
    """
    msg_vec  = torch.from_numpy(embedded_user_chat_logs).unsqueeze(0)
    key_vecs = torch.from_numpy(__EMBEDS)

    msg_vec  = F.normalize(msg_vec,  dim=1) # L2 정규화
    key_vecs = F.normalize(key_vecs, dim=1)

    sims = torch.mm(msg_vec, key_vecs.T).squeeze(0) # (k,)

    top_scores, idx = torch.topk(sims, k=min(top_k, sims.size(0)))
    return [__KEYS[i] for j, i in idx if top_scores[j] >= min_sim]
