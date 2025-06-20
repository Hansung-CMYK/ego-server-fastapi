import re


class TTSBuffer:
    def __init__(self, send_tts_fn, min_length=8):
        self.buffer = ""
        self.send_tts = send_tts_fn
        self.min_length = min_length

    def _avoid_leading_number(self, text: str) -> str:
        if re.match(r'^\s*\d', text):
            return '\u200B' + text
        return text

    def feed(self, chunk: str):
        self.buffer += chunk
        if re.search(r"[,\.\?\!~:]+", chunk):
            text = self.buffer.strip()
            if re.fullmatch(r"[,\.\?\!~:]+", text):
                self.buffer = ""
                return
            if len(text) >= self.min_length:
                # 맨 앞 숫자 피하기
                text = self._avoid_leading_number(text)
                self.send_tts(text)
                self.buffer = ""

    def flush(self):
        text = self.buffer.strip()
        if text:
            text = self._avoid_leading_number(text)
            self.send_tts(text)
        self.buffer = ""
