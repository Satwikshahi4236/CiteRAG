from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
  client = TestClient(app)
  resp = client.get("/health")
  assert resp.status_code == 200
  data = resp.json()
  assert "status" in data

