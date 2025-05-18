from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

class DailyCommentLLM:
    def __init__(self):
        model = ChatOllama(
            model="gemma3",
            temperature=0.7
        )
        prompt = ChatPromptTemplate.from_messages(self.__DAILY_TEMPLATE)
        self.__daily_chain = prompt | model

    def invoke(self, diaries:list[dict] , feeling:list[str], keywords:list[str]):
        events = [diary[""] for diary in diaries]
        return self.__daily_chain.invoke({"events": events, "feeling": feeling, "keywords":keywords}).content.strip()

    __DAILY_TEMPLATE = [
        ("system", "/no_think\n"),
        ("system", "너는 부드럽게 바꿔주는 사람이다."),
        ("system", "{events}가 있었던 오늘, {feelings} 감정이 하루를 지배했고, {keywords}가 곁을 맴돌았어요."),
    ]

daily_comment_llm = DailyCommentLLM()
