from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_safe_demo_status():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["data_mode"] == "synthetic-only"
    assert body["payment_actions_enabled"] is False
    assert body["environment"] == "local-demo"
