from textwrap import dedent

from config.common.common_llm import CommonLLM


class SummaryLLM(CommonLLM):

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
        return super().invoke(parameter)
