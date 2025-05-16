import fal_client
from langchain_ollama import OllamaLLM

TEMPLATE = """
You are an expert at transforming arbitrary text descriptions into vivid, detailed English prompts for image generation.
Extract the key visual elements, style, lighting, colors, and composition from the user’s text.
Output MUST be a single, concise English sentence or two, focusing purely on the visual prompt—no extra commentary.

Example:
Input: “한강에서 밤에 반짝이는 불빛과 사람들이 노는 모습”
Output: “A vibrant nighttime scene on the Han River, glowing city lights reflecting on water, silhouettes of people playing and relaxing on the riverbank, soft ambient lighting and rich color contrast.”

Now generate the image prompt for this text:

“{user_text}”
"""

def generate_image_prompt(user_text: str) -> str:
    ollama = OllamaLLM(model="gemma3:4b")
    
    prompt = TEMPLATE.format(user_text=user_text)
    
    output = ollama.invoke(prompt)
    return output

def generate_image(prompt: str) -> str :
    # NOTE: 추후 삭제 (로그)
    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(log["message"])

    result = fal_client.subscribe(
        "fal-ai/flux-lora",
        arguments={
            "prompt": prompt
        },
        with_logs=True,
        on_queue_update=on_queue_update,
    )

    return result
