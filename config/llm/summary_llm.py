from textwrap import dedent

from langchain_core.prompts import ChatPromptTemplate

from config.common.default_model import llm_sem, clean_json_string


class SummaryLlm:
    def __init__(self):
        prompt = ChatPromptTemplate.from_messages(self.__SUMMARY_TEMPLATE)
        self.__chain = prompt | chat_model

    def summary_invoke(self, chat_rooms: list[str]) -> str:
        with llm_sem:
            answer = self.__chain.invoke({
                "input": "\n".join(chat_rooms),
            }).content
        answer = clean_json_string(answer)
        logger.info(msg=f"\n\nPOST: api/v1/diary [summary_invoke 요약문]\n{answer}\n")
        return answer

    __SUMMARY_TEMPLATE = [
        ("system", "/no_think"),
        ("system", dedent("""
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
    ]

summary_llm = SummaryLlm()
