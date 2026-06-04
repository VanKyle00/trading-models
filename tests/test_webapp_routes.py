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


def test_index_lists_models_and_params():
    client = TestClient(create_app())
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "SMA Crossover on SPY" in html  # a model name from ModelSpec
    assert "Fast SMA" in html  # a param label from ModelSpec


def test_index_has_theme_toggle_and_assistant_console():
    client = TestClient(create_app())
    html = client.get("/").text
    assert 'data-theme-set="bone"' in html and 'data-theme-set="night"' in html
    assert 'id="composer"' in html  # the assistant console input
    assert "/api/v1/chat" in html  # console wired to the SSE endpoint


def test_run_partial_renders_metric_strip_and_run_context():
    client = TestClient(create_app())
    resp = client.post(
        "/run",
        data={
            "model_id": "models/classical/01-sma-crossover-spy",
            "symbol": "SPY",
            "start": "2022-01-01",
            "end": "2023-01-01",
            "fast": "20",
            "slow": "50",
        },
    )
    assert resp.status_code == 200
    html = resp.text
    assert "metric-strip" in html  # the hero metrics grid
    assert 'id="run-context"' in html  # context blob the console reads
    assert "Max Drawdown" in html  # a hero metric label


def test_run_partial_renders_results():
    client = TestClient(create_app())
    resp = client.post(
        "/run",
        data={
            "model_id": "models/classical/01-sma-crossover-spy",
            "symbol": "SPY",
            "start": "2022-01-01",
            "end": "2023-01-01",
            "fast": "20",
            "slow": "50",
        },
    )
    assert resp.status_code == 200
    assert "plotly" in resp.text.lower()  # an embedded figure
    assert "sharpe" in resp.text.lower()  # a metric


def test_run_partial_bad_request_shows_error():
    client = TestClient(create_app())
    resp = client.post(
        "/run",
        data={
            "model_id": "models/classical/01-sma-crossover-spy",
            "start": "2023-01-01",
            "end": "2022-01-01",
        },
    )
    assert resp.status_code == 400
    assert "before end" in resp.text or "error" in resp.text.lower()


def test_run_partial_bad_ticker_shows_error_not_500():
    client = TestClient(create_app(), raise_server_exceptions=False)
    resp = client.post(
        "/run",
        data={
            "model_id": "models/classical/01-sma-crossover-spy",
            "symbol": "NOTAREALTICKER_ZZZ",
            "start": "2022-01-01",
            "end": "2023-01-01",
            "fast": "20",
            "slow": "50",
        },
    )
    assert resp.status_code == 400
    assert "could not run" in resp.text.lower()
