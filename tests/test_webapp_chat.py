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
