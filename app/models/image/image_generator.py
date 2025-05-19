import fal_client
from langchain_core.messages import HumanMessage

from app.models.default_model import chat_model

class ImageGenerator:
    @staticmethod
    def generate(prompt_ko: str) -> str:

        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    print(log["message"])

        human_msg = HumanMessage(
            f"""
                You are an expert at transforming arbitrary text descriptions into vivid, detailed English prompts for image generation.
                Extract the key visual elements, style, lighting, colors, and composition from the user’s text.
                Output MUST be a single, concise English sentence or two, focusing purely on the visual prompt—no extra commentary.

                Example:
                Input: “한강에서 밤에 반짝이는 불빛과 사람들이 노는 모습”
                Output: “A vibrant nighttime scene on the Han River, glowing city lights reflecting on water, silhouettes of people playing and relaxing on the riverbank, soft ambient lighting and rich color contrast.”

                Now generate the image prompt for this text:

                “{prompt_ko}”
            """
        )
        prompt_eng = chat_model.invoke([human_msg]).content

        result = fal_client.subscribe(
            "fal-ai/flux-lora",
            arguments={
                "prompt": prompt_eng
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )

        try:
            images = result.get("images")
            if not images or not isinstance(images, list) or "url" not in images[0]:
                raise ValueError("Invalid image result format")
            return images[0]["url"]
        except Exception as e:
            return "이미지 생성 오류"