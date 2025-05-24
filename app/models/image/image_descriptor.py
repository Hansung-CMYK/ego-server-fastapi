import base64

from app.services.session_config import SessionConfig
from langchain_core.messages import HumanMessage
from app.models.default_model import chat_model as vision_model
from app.models.chat.main_llm import main_llm

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
