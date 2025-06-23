import base64

from langchain_core.messages import HumanMessage

from config.common.common_llm import chat_model as vision_model
from config.llm.main_llm import main_llm
from config.session.session_config import SessionConfig


class ImageDescriptor:
    @staticmethod
    def invoke(b64_image: base64) -> str:
        human_msg = HumanMessage(
            content=[
                {"type": "image_url", "image_url": b64_image},
                {"type": "text",      "text": "What's this? Provide a description in korean without leading or trailing text or markdown syntax."}
            ]
        )
        return vision_model.invoke([human_msg]).content

    @staticmethod
    def store(image_description : str, session_config : SessionConfig):
        main_llm.add_message_in_session_history(session_id=session_config.session_id, human_message=image_description)
