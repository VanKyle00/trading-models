# tests/test_webapp_routes.py
from fastapi.testclient import TestClient

from webapp.main import create_app


def test_healthz_ok():
    client = TestClient(create_app())
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_api_run_bad_request_is_400():
    client = TestClient(create_app())
    resp = client.post(
        "/api/v1/run",
        json={
            "model_id": "models/classical/01-sma-crossover-spy",
            "start": "2023-01-01",
            "end": "2022-01-01",  # reversed → RequestError
        },
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_api_run_unknown_model_is_400():
    client = TestClient(create_app())
    resp = client.post(
        "/api/v1/run",
        json={
            "model_id": "models/nope/x",
            "start": "2022-01-01",
            "end": "2023-01-01",
        },
    )
    assert resp.status_code == 400


def test_api_run_success_returns_baseline_keys():
    client = TestClient(create_app())
    resp = client.post(
        "/api/v1/run",
        json={
            "model_id": "models/classical/01-sma-crossover-spy",
            "symbol": "SPY",
            "start": "2022-01-01",
            "end": "2023-01-01",
            "fast": 20,
            "slow": 50,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert {"model_id", "symbol", "metrics", "series", "trades"} <= set(body)


def test_api_run_non_object_json_is_400():
    client = TestClient(create_app())
    resp = client.post("/api/v1/run", json=[1, 2, 3])
    assert resp.status_code == 400


def test_api_run_invalid_json_is_400():
    client = TestClient(create_app())
    resp = client.post(
        "/api/v1/run", content=b"not json", headers={"content-type": "application/json"}
    )
    assert resp.status_code == 400
