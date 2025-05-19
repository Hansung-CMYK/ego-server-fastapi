import numpy as np
import torch
from app.models.embedding_model import embedding_model

class PreferenceModel:
    __keywords = ["매력적", "즐거운", "만족한", "원만한", "지루한", "불안한", "부정적"]

    __preference_embeddings = torch.tensor(np.load("preference_embeddings.npy"))

    def classify_sentence(self, sentence: str, threshold: float = 0.3) -> str:
        sentence_embedding = embedding_model.embed_documents([sentence])[0]
        sentence_embedding = torch.tensor(sentence_embedding).unsqueeze(0)

        cos_scores = torch.nn.functional.cosine_similarity(sentence_embedding, self.__preference_embeddings)
        max_score, max_idx = torch.max(cos_scores, dim=0)

        if max_score.item() < threshold:
            return "없음"
        return self.__keywords[max_idx.item()]

prefence_model = PreferenceModel()

# 테스트 코드
def save():
    perference = ["매력적", "즐거운", "만족한", "원만한", "지루한", "불안한", "부정적"]
    # perference = [
    #     # 매력적
    #     "매력적이다. 사람의 마음을 사로잡아 끄는 힘이 있는. 매력적 몸매. 그는 매력적 조건을 제시했다. 매력적인 웃음. 그 가수의 목소리는 매력적이다. 큰 눈에 높직한 코와 뚜렷한 윤곽이 아주 매력적이었다.",
    #     # 즐거운
    #     "마음에 거슬림이 없이 흐뭇하고 기쁘다. 즐거운 여행. 즐겁게 지내다. 나는 요즘 하루하루가 즐겁다. 우리는 함께 지내는 것이 즐겁기만 하였다. 문태석은 경채를 만나기가 즐거웠다.",
    #     # 만족한
    #     "마음에 흡족하다. 모자람이 없이 넉넉하다. 만족한 얼굴. 만족한 웃음. 만족한 성과를 올리다. 형사는 그에게서 만족하게 그 질문에 대한 대답을 듣지 못했다. 선생은 그것을 바라보고 만족한 듯하다.",
    #     # 원만한
    #     "일의 진행이 순조롭다. 서로 사이가 좋다. 원만한 부부 생활. 원만한 관계를 유지하다. 대인 관계를 원만하게 하다. 민지는 너무 이기적이어서 친구들과 원만하게 지내지 못한다. 연수는 시어머니와 관계가 원만하지 못했다.",
    #     # 지루한
    #     "시간이 오래 걸리거나 같은 상태가 오래 계속되어 따분하고 싫증이 나다. 영화가 지루하다. 밤은 무덥고 지루했다. 논쟁이 지루하게 계속되었다. 나는 기다리는 것이 지루하여 옆에 있는 잡지를 뒤적거리기 시작했다. 임명빈은 어눌한 말씨로 지루하게 설명을 하는 것이었다.",
    #     # 불안한
    #     "마음이 편하지 아니하다. 분위기 따위가 술렁거리어 뒤숭숭하다. 서동수는 이쪽이 다섯밖에 안 되는 것이 갑자기 좀 불안했다. 서울이 처음인 모양인 또 한 사람은 몹시 불안한 표정으로 남산을 올려다본다 나는 집에 혼자 있기가 불안하여 친구를 불렀다. 국제 경제 정세가 불안하면 국내 경제가 위축되곤 하였다. 이 시끄럽고 불안한 시국에 입단속 잘못했다간 엉뚱한 화를 입게 될지도 모르니까.",
    #     # 부정적
    #     "그렇지 아니하다고 단정하거나 옳지 아니하다고 반대하는. 바람직하지 못한. 부정적 상황. 부정적 이미지. 옛날과 달리 아버지의 모든 것을 조금씩 부정적인 것으로 보기 시작했다. 불안과 고독은 인생의 어두운 측면이요, 부정적 요소이다. 대중 매체의 부정적인 면을 강조하다. 학생들에게 부정적인 영향을 주다."
    # ]

    import numpy as np
    from app.models.embedding_model import embedding_model
    embeddings = embedding_model.embed_documents(perference)
    embedding_array = np.stack(embeddings)

    # 저장 파일명 (timestamp 포함)
    filename = f"preference_embeddings.npy"
    np.save(filename, embedding_array)

    print(f"✅ 임베딩을 '{filename}' 파일로 저장했습니다.")

# save()

def main():
    reuslt = prefence_model.classify_sentence("""
        AI: 안녕 오늘 무슨 일이 있었어?
        Human: 야이 바보야 너랑 이야기 하면 꼭 화만 나!
        AI: 무슨 일인데 그래?
        Human: 내가 계속 너한테 존댓말 쓰지 말라고 하는데, 계속하면 답답해 안답답해
    """)
    print(reuslt)

if __name__ == "__main__":
    main()