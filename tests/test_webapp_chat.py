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


def test_chat_rejects_empty_message():
    client = TestClient(create_app())
    resp = client.post("/api/v1/chat", json={"message": ""})
    assert resp.status_code == 400
