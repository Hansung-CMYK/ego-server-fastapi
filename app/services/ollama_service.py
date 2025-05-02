import ollama

try:
    ollama.pull("gemma3:4b")
except Exception:
    pass

def chat_full(prompt: str, model: str = "gemma3:4b") -> str:
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp["message"]["content"]

def chat_stream(prompt: str, model: str = "gemma3:4b"):
    stream = ollama.generate(
        model=model,
        prompt=prompt,
        stream=True
    )
    for chunk in stream:
        yield chunk["response"]

