from pprint import pprint

from fastapi.testclient import TestClient

from app.internal.admin.dto.admin_request import AdminRequest, ADMIN_PASSWORD, ADMIN_ID
from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.post("/api/admin/test", json=AdminRequest(admin_id=ADMIN_ID, admin_password=ADMIN_PASSWORD).model_dump())
    print("여기서 부터 pprint")
    pprint(response.json())
    print("여기서 까지 pprint")
    assert response.json() == {
        "code":200,
        'data': None,
        "message":"test success"
    }