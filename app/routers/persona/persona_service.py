from app.routers.persona.persona_repository import PersonaRepository

persona_repository = PersonaRepository()

def insert_persona(ego_id: str, persona: dict):
    return persona_repository.insert_persona(ego_id=ego_id, persona=persona)

def select_persona_to_ego_id(ego_id: str)->tuple:
    return persona_repository.select_persona_to_ego_id(ego_id=ego_id)

def update_persona(ego_id: str, persona: dict):
    return persona_repository.update_persona(ego_id=ego_id, persona=persona)

def delete_persona(ego_id: str):
    return persona_repository.delete_persona(ego_id=ego_id)

def has_persona(ego_id: str) -> bool:
    return persona_repository.has_persona(ego_id=ego_id)
