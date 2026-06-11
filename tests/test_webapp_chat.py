# tests/test_webapp_chat.py
import json

from fastapi.testclient import TestClient

from tradinglib.assistant.types import AssistantTurn, Usage
from webapp.main import create_app


def _sse_events(text: str):
    out = []
    for line in text.splitlines():
        if line.startswith("data:"):
            out.append(json.loads(line[len("data:") :].strip()))
    return out


def test_chat_streams_final_event_with_stub(monkeypatch):
    from tradinglib.assistant import provider as provider_mod

    scripted = [
        AssistantTurn(text="Five models.", tool_calls=(), stop_reason="end_turn", usage=Usage(3, 3))
    ]
    monkeypatch.setattr(
        provider_mod,
        "ClaudeProvider",
        lambda *a, **k: provider_mod.StubProvider(list(scripted)),
    )

    client = TestClient(create_app())
    resp = client.post("/api/v1/chat", json={"message": "how many models?"})
    assert resp.status_code == 200
    events = _sse_events(resp.text)
    assert events[-1]["type"] == "final"
    assert "Five" in events[-1]["text"]


def test_chat_forwards_onscreen_context_to_agent(monkeypatch):
    from tradinglib.assistant import provider as provider_mod

    scripted = [
        AssistantTurn(text="done", tool_calls=(), stop_reason="end_turn", usage=Usage(3, 3))
    ]
    stub = provider_mod.StubProvider(list(scripted))
    monkeypatch.setattr(provider_mod, "ClaudeProvider", lambda *a, **k: stub)

    client = TestClient(create_app())
    resp = client.post(
        "/api/v1/chat",
        json={
            "message": "explain the worst month",
            "context": {
                "model": "Momentum",
                "symbol": "SPY",
                "start": "2022-01-01",
                "end": "2024-12-31",
                "metrics": {"Sharpe": 1.42, "Max Drawdown": -0.081},
            },
        },
    )
    assert resp.status_code == 200
    opening = stub.calls[0][0].text  # the agent's first UserMsg
    assert "SPY" in opening and "Sharpe=1.42" in opening
    assert "explain the worst month" in opening


def test_chat_rejects_empty_message():
    client = TestClient(create_app())
    resp = client.post("/api/v1/chat", json={"message": ""})
    assert resp.status_code == 400


def test_chat_provider_failure_still_emits_final(monkeypatch):
    from tradinglib.assistant import provider as provider_mod

    def boom(*a, **k):
        raise RuntimeError("no API key configured")

    monkeypatch.setattr(provider_mod, "ClaudeProvider", boom)

    client = TestClient(create_app())
    resp = client.post("/api/v1/chat", json={"message": "hello"})
    assert resp.status_code == 200  # stream already opened
    events = _sse_events(resp.text)
    assert events[-1]["type"] == "final"  # graceful terminal event, not a truncated stream
    assert "unavailable" in events[-1]["text"].lower()


def test_chat_provider_local_uses_local_adapter(monkeypatch):
    # provider="local" must route to the fine-tuned adapter, not Claude.
    import webapp.main as main
    from tradinglib.assistant import provider as provider_mod

    def claude_boom(*a, **k):
        raise AssertionError("local request must not construct ClaudeProvider")

    monkeypatch.setattr(provider_mod, "ClaudeProvider", claude_boom)
    scripted = [
        AssistantTurn(
            text="Local says hi.", tool_calls=(), stop_reason="end_turn", usage=Usage(3, 3)
        )
    ]
    monkeypatch.setattr(
        main, "_get_local_provider", lambda: provider_mod.StubProvider(list(scripted))
    )

    client = TestClient(create_app())
    resp = client.post("/api/v1/chat", json={"message": "hi", "provider": "local"})
    assert resp.status_code == 200
    events = _sse_events(resp.text)
    assert events[-1]["type"] == "final"
    assert "Local" in events[-1]["text"]


def test_chat_local_load_failure_emits_helpful_final(monkeypatch):
    # If the adapter can't load (no GPU / missing weights), tell the user to switch.
    import webapp.main as main

    def boom():
        raise RuntimeError("CUDA not available")

    monkeypatch.setattr(main, "_get_local_provider", boom)

    client = TestClient(create_app())
    resp = client.post("/api/v1/chat", json={"message": "hi", "provider": "local"})
    assert resp.status_code == 200
    events = _sse_events(resp.text)
    assert events[-1]["type"] == "final"
    assert "local model unavailable" in events[-1]["text"].lower()


# ── conversation history ───────────────────────────────────────────────


def _stub_chat_provider(monkeypatch):
    from tradinglib.assistant import provider as provider_mod

    scripted = [
        AssistantTurn(text="noted", tool_calls=(), stop_reason="end_turn", usage=Usage(3, 3))
    ]
    stub = provider_mod.StubProvider(list(scripted))
    monkeypatch.setattr(provider_mod, "ClaudeProvider", lambda *a, **k: stub)
    return stub


def test_chat_forwards_history_to_agent(monkeypatch):
    stub = _stub_chat_provider(monkeypatch)

    client = TestClient(create_app())
    resp = client.post(
        "/api/v1/chat",
        json={
            "message": "use those levels",
            "history": [
                {"role": "user", "text": "I'm bullish on RIVN"},
                {"role": "assistant", "text": "Proposed: entry 14.2, stop 12.9"},
            ],
        },
    )
    assert resp.status_code == 200
    msgs = stub.calls[0]
    assert msgs[0].text == "I'm bullish on RIVN"
    assert msgs[1].turn.text == "Proposed: entry 14.2, stop 12.9"
    assert msgs[2].text == "use those levels"


def test_chat_rejects_malformed_history(monkeypatch):
    _stub_chat_provider(monkeypatch)
    client = TestClient(create_app())

    for bad in (
        "not a list",
        [42],
        [{"role": "system", "text": "x"}],
        [{"text": "no role"}],
    ):
        resp = client.post("/api/v1/chat", json={"message": "hi", "history": bad})
        assert resp.status_code == 400, f"accepted {bad!r}"


def test_chat_history_absent_or_null_is_fine(monkeypatch):
    _stub_chat_provider(monkeypatch)
    client = TestClient(create_app())
    resp = client.post("/api/v1/chat", json={"message": "hi", "history": None})
    assert resp.status_code == 200


def test_chat_history_caps_messages_and_chars(monkeypatch):
    stub = _stub_chat_provider(monkeypatch)
    client = TestClient(create_app())

    long_history = []
    for i in range(30):  # alternating, starts with user; only the last 20 survive
        role = "user" if i % 2 == 0 else "assistant"
        long_history.append({"role": role, "text": f"m{i}" + "x" * 5000})
    resp = client.post("/api/v1/chat", json={"message": "tail", "history": long_history})
    assert resp.status_code == 200
    msgs = stub.calls[0]
    assert len(msgs) == 21  # 20 history (alternating, no merging) + opening
    assert msgs[0].text.startswith("m10")  # the cap keeps the most recent 20
    assert all(
        len(m.text if hasattr(m, "text") else m.turn.text) <= 4000 + len("tail") + 2 for m in msgs
    )


# ── planner sizing settings ────────────────────────────────────────────


def test_chat_forwards_planner_settings_to_agent(monkeypatch):
    stub = _stub_chat_provider(monkeypatch)
    client = TestClient(create_app())
    resp = client.post(
        "/api/v1/chat",
        json={
            "message": "I'm bullish on RIVN",
            "settings": {"account_size": 50000, "risk_per_trade_pct": 0.02},
        },
    )
    assert resp.status_code == 200
    opening = stub.calls[0][0].text
    assert "account size $50,000" in opening
    assert "2%" in opening
    assert "do not ask the user for account size or risk" in opening


def test_chat_malformed_settings_ignored_not_400(monkeypatch):
    # Bad settings degrade to the no-settings flow — never block the chat.
    client = TestClient(create_app())
    for bad in (
        "not a dict",
        {"account_size": -5, "risk_per_trade_pct": 0.01},
        {"account_size": 100000, "risk_per_trade_pct": 0.5},
        {"account_size": "lots"},
        {"account_size": "inf", "risk_per_trade_pct": 0.02},
        {"account_size": "nan", "risk_per_trade_pct": 0.02},
        None,
    ):
        stub = _stub_chat_provider(monkeypatch)
        resp = client.post("/api/v1/chat", json={"message": "hi", "settings": bad})
        assert resp.status_code == 200, f"rejected {bad!r}"
        assert "Planner sizing" not in stub.calls[0][0].text, f"accepted {bad!r}"
