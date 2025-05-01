from app.services.tts_model_registry import get_model
from app.services.tts_infer          import get_tts_wav

def synthesize(
    model_id: str,
    ref_wav_path: str,
    prompt_text: str,
    prompt_language: str,
    text: str,
    text_language: str,
    **kwargs
):
    speaker = get_model(model_id)           # Speaker 객체
    return get_tts_wav(                     # GPT-SoVITS 원본 함수
        ref_wav_path=ref_wav_path,
        prompt_text=prompt_text,
        prompt_language=prompt_language,
        text=text,
        text_language=text_language,
        spk=speaker.name,                   # speaker_list 키
        **kwargs
    )

