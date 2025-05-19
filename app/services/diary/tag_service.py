from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from app.models.default_model import kiwi
from typing import List, Tuple

from app.services.diary.tag_embedding_service import load_index

_KEYS, _EMBEDS = load_index()


def _sentence_embedding(text: str) -> np.ndarray:
    from app.models.normalization.embedding_model import embedding_model
    # 명사만 추출해 노이즈 감소
    nouns = [
        tok.form
        for sent in kiwi.analyze(text)
        for tok in sent[0]
        if tok.tag.startswith("NN")
    ]
    message = " ".join(nouns) if nouns else text
    return embedding_model.embeded_documents([message])[0].astype(np.float32)


def rank_keywords(
    text: str,
    top_k: int = 5,
    min_sim: float = 0.30,
) -> List[str]:
    """
    Parameters:
        text     : 분석할 문장
        top_k    : 상위 k개 후보 반환 (기본 5)
        min_sim  : 코사인 유사도 컷오프 (기본 0.30)

    Returns:
        List[<keyword>]
    """
    msg_vec  = torch.from_numpy(_sentence_embedding(text)).unsqueeze(0)  # (1, dim)
    key_vecs = torch.from_numpy(_EMBEDS)                                 # (k, dim)

    # L2 정규화
    msg_vec  = F.normalize(msg_vec,  dim=1)
    key_vecs = F.normalize(key_vecs, dim=1)

    sims = torch.mm(msg_vec, key_vecs.T).squeeze(0)    # (k,)

    top_scores, idx = torch.topk(sims, k=min(top_k, sims.size(0)))
    return [_KEYS[i] for j, i in idx if top_scores[j] >= min_sim]
