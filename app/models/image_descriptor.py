import base64

from langchain_core.messages import HumanMessage
from app.models.main_llm import chat_model

class ImageDescriptor:
    def __init__(self):
        self.__vision_model = chat_model

    def invoke(self, b64_image: base64) -> str:
        human_msg = HumanMessage(
            content=[
                {"type": "image_url", "image_url": b64_image},
                {"type": "text",      "text": "What's this? Provide a description in korean without leading or trailing text or markdown syntax."}
            ]
        )
        return self.__vision_model.invoke([human_msg]).content
