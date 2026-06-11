# tests/test_webapp_routes.py
from fastapi.testclient import TestClient

from tradinglib.service import list_specs
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


def test_index_ticker_control_honors_lock():
    client = TestClient(create_app())
    html = client.get("/").text
    # exactly one `symbol` field so the form can never submit duplicate keys
    assert html.count('name="symbol"') == 1
    # each model option carries the ticker metadata the JS needs to drive it
    assert "data-ticker-mode=" in html
    assert html.count("data-ticker-default=") == len(list_specs())
    # the locked models expose their fixed mode + locked symbol to the JS
    assert 'data-ticker-mode="fixed"' in html
    for locked in ("BTC-USD", "BTCUSDT", "SPY"):
        assert f'data-ticker-default="{locked}"' in html


def test_index_has_model_aware_event_presets():
    client = TestClient(create_app())
    html = client.get("/").text
    assert "Market event" in html
    assert "data-model-events=" in html
    # equities model surfaces an equity event, crypto model a crypto event
    assert "GFC 2008" in html
    assert "FTX Collapse" in html
    # a preset carries its date range for the JS to apply
    assert 'data-start="2020-02-19"' in html


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


def test_index_console_extraction_preserves_markup():
    client = TestClient(create_app())
    html = client.get("/").text
    assert 'id="console"' in html
    assert 'id="console-resize"' in html  # index keeps the resize handle
    assert 'id="chat-input"' in html and 'id="chat-clear"' in html
    assert "Run a backtest, then ask about it" in html  # default empty-state text
    assert html.count('id="transcript"') == 1


def test_console_ships_ticket_card_renderer():
    client = TestClient(create_app())
    html = client.get("/").text
    assert "renderTicketCard" in html
    assert "build_options_ticket" in html  # the tool_result hook


def test_planner_page_renders_console_and_chips():
    client = TestClient(create_app())
    resp = client.get("/planner")
    assert resp.status_code == 200
    html = resp.text
    assert 'id="transcript"' in html and 'id="chat-input"' in html
    assert 'class="chat-chip"' in html and "bullish on RIVN" in html  # chip BUTTONS render
    assert "renderTicketCard" in html
    assert 'id="console-resize"' not in html  # resize handle is index-only
    assert html.count('id="transcript"') == 1  # the partial is included exactly once
    assert 'href="/scans"' in html and 'href="/tournaments"' in html and 'href="/models"' in html


def test_planner_nav_crumbs():
    client = TestClient(create_app())
    assert 'href="/planner"' in client.get("/").text
    assert 'href="/planner"' in client.get("/models").text


def test_index_has_no_chip_buttons():
    # The partial's CSS/JS mention chat-chip on every page; only the planner
    # context (console_chips set) renders actual chip BUTTONS.
    client = TestClient(create_app())
    assert 'class="chat-chip"' not in client.get("/").text
