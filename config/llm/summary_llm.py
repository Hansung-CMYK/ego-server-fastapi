from textwrap import dedent

from config.common.common_llm import CommonLLM


class SummaryLLM(CommonLLM):
    """
    요약:
        대화 내역을 요약하는 LLM

    설명:
        chat-history Table의 대화 내역의 불필요한 내용을 제거하고 요약한다.
        대화 내역이 많을 시, LLM의 프롬프트가 가려지는 문제가 발생해 이를 해결하기 위해 적용하였다.

    Attributes:
        __SUMMARY_TEMPLATE(tuple): 대화 내역을 요약하기 위한 시스템 프롬프트
    """

    __SUMMARY_TEMPLATE = ("system", dedent("""
        You are an assistant who refines raw chat history to help another AI diarist understand a user's day.

        <SUMMARY_GUIDELINES>
        - Your job is to condense long chat logs into the minimal but complete version.
        - Eliminate duplicated or repeated messages.
        - Do not add emotional interpretation or inference.
        - Focus on the clearest version of what actually happened.
        - Preserve only user-facing lines (e.g., u@ lines), trimming unrelated system or bot lines.
        - Keep the refined log as concise as possible while covering the essential interactions.
        </SUMMARY_GUIDELINES>

        <CHAT_LOG>
        {input}
        </CHAT_LOG>
        """).strip())

    def __add_template(self)->list[tuple]:
        return [self.__SUMMARY_TEMPLATE]

    def invoke(self, parameter:dict)->str:
        """
        요약:
            대화내역에서 불필요한 내용을 제거하고 요약하는 함수

        Parameters:
            parameter(dict): parameter는 다음과 같은 key-value를 갖는다.
                - input(str): 채팅방 대화 내역(하나의 문자열로 된 대화내역)
        """
        return super().invoke(parameter)
