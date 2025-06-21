from app.internal.tone.tone_repository import ToneRepository

tone_repository = ToneRepository()

def delete_tone(ego_id: str):
    return tone_repository.delete_tone(ego_id=ego_id)