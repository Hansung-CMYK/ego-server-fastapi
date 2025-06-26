from fastapi.testclient import TestClient

from app.internal.admin.dto.admin_request import (ADMIN_ID, ADMIN_PASSWORD,
                                                  AdminRequest)
from app.main import app

client = TestClient(app)
admin_request_json = AdminRequest(admin_id=ADMIN_ID, admin_password=ADMIN_PASSWORD).model_dump()

def test():
    response = client.post("/api/admin/test", json=admin_request_json)
    assert response.json() == {
        "code":200,
        'data': None,
        "message":"test success"
    }

def test_reset_ego():
    response = client.post("/api/admin/reset/user_id_001/1", json=admin_request_json)