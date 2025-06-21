from app.routers.tone.tone_repository import ToneRepository

tone_repository = ToneRepository()

def insert_tone(ego_id: str, tone: dict):
    return tone_repository.insert_tone(ego_id=ego_id, tone=tone)

def delete_tone(ego_id: str):
    return tone_repository.delete_tone(ego_id=ego_id)

def has_tone(ego_id: str) -> bool:
    return tone_repository.has_tone(ego_id=ego_id)