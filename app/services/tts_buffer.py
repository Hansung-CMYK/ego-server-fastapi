import re

class TTSBuffer:
    def __init__(self, send_tts_fn, min_length=8):
        self.buffer = ""
        self.send_tts = send_tts_fn
        self.min_length = min_length

    def feed(self, chunk: str):
        self.buffer += chunk
        # 청크에 문장부호가 포함되면 검사
        if re.search(r"[,\.\?\!~:]+", chunk):
            text = self.buffer.strip()
            # 문장부호만 남아있으면 초기화
            if re.fullmatch(r"[,\.\?\!~:]+", text):
                self.buffer = ""
                return
            # 충분히 길면 flush
            if len(text) >= self.min_length:
                text = re.sub(r"^[,\.\?\!~:]+", "", text).strip()
                if text:
                    self.send_tts(text)
                self.buffer = ""

    def flush(self):
        text = self.buffer.strip()
        if text:
            text = re.sub(r"^[,\.\?\!~:]+", "", text).strip()
            if text:
                self.send_tts(text)
        self.buffer = ""
