# tests/test_assistant_tools.py
import json

from tradinglib.assistant.tools import TOOL_SPECS, dispatch


def test_tool_specs_are_the_five_assistant_tools():
    names = {t["name"] for t in TOOL_SPECS}
    assert names == {
        "list_models",
        "get_model_spec",
        "run_backtest",
        "propose_trade_levels",
        "build_options_ticket",
    }
    for t in TOOL_SPECS:
        assert "description" in t and "input_schema" in t


def test_list_models_returns_models():
    out, is_error = dispatch("list_models", {})
    assert is_error is False
    models = json.loads(out)["models"]
    assert any(m["family"] == "classical" for m in models)


def test_get_model_spec_returns_params():
    out, is_error = dispatch(
        "get_model_spec", {"model_id": "models/classical/01-sma-crossover-spy"}
    )
    assert is_error is False
    spec = json.loads(out)
    assert spec["ticker_mode"] == "free"
    assert {"fast", "slow"} <= {p["name"] for p in spec["params"]}


def test_get_model_spec_unknown_is_error():
    out, is_error = dispatch("get_model_spec", {"model_id": "nope"})
    assert is_error is True
    assert "nope" in out


def test_run_backtest_invalid_config_is_error_not_crash():
    out, is_error = dispatch(
        "run_backtest",
        {
            "model_id": "models/classical/01-sma-crossover-spy",
            "start": "2023-01-01",
            "end": "2022-01-01",  # reversed
        },
    )
    assert is_error is True
    assert "start" in out.lower()


def test_unknown_tool_is_error():
    _out, is_error = dispatch("frobnicate", {})
    assert is_error is True


def test_run_backtest_happy_path_returns_compact_summary():
    out, is_error = dispatch(
        "run_backtest",
        {
            "model_id": "models/classical/01-sma-crossover-spy",
            "symbol": "SPY",
            "start": "2022-01-01",
            "end": "2023-01-01",
            "params": {"fast": 20, "slow": 50},
        },
    )
    assert is_error is False
    payload = json.loads(out)
    assert "metrics" in payload and "series" not in payload  # compact, no full series
    assert isinstance(payload["n_trades"], int)


def test_dispatch_never_raises_on_internal_error(monkeypatch):
    # Force an unexpected (non-Request/Key/Value) error inside a handler and
    # confirm dispatch converts it to an is_error result instead of propagating.
    import tradinglib.assistant.tools as tools_mod

    def boom(*a, **k):
        raise ConnectionError("network down")

    monkeypatch.setattr(tools_mod, "list_specs", boom)
    out, is_error = dispatch("list_models", {})
    assert is_error is True
    assert "list_models" in out


# ── options-planner tools ──────────────────────────────────────────────


def test_propose_trade_levels_happy_path(monkeypatch):
    import numpy as np
    import pandas as pd

    import tradinglib.assistant.planner as planner

    close = 100.0 + np.sin(np.arange(300) / 5.0)
    idx = pd.date_range("2025-06-01", periods=300, freq="B")
    bars = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6},
        index=idx,
    )
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)

    out, is_error = dispatch("propose_trade_levels", {"ticker": "rivn", "stance": "LONG"})

    assert is_error is False
    payload = json.loads(out)
    assert payload["ticker"] == "RIVN"
    assert payload["levels"]["stop"] < payload["levels"]["entry"] < payload["levels"]["target"]


def test_propose_trade_levels_rejects_bad_stance():
    out, is_error = dispatch("propose_trade_levels", {"ticker": "RIVN", "stance": "sideways"})
    assert is_error is True
    assert "stance" in out


def test_propose_trade_levels_requires_ticker():
    out, is_error = dispatch("propose_trade_levels", {"stance": "long"})
    assert is_error is True
    assert "ticker" in out


def test_build_options_ticket_rejects_bad_long_geometry():
    out, is_error = dispatch(
        "build_options_ticket",
        {
            "ticker": "RIVN",
            "stance": "long",
            "entry": 100.0,
            "stop": 108.0,  # stop above entry on a long
            "target": 96.0,
            "account_size": 100_000.0,
            "risk_per_trade_pct": 0.01,
        },
    )
    assert is_error is True
    assert "stop < entry < target" in out


def test_build_options_ticket_rejects_percent_style_risk():
    out, is_error = dispatch(
        "build_options_ticket",
        {
            "ticker": "RIVN",
            "stance": "long",
            "entry": 100.0,
            "stop": 96.0,
            "target": 108.0,
            "account_size": 100_000.0,
            "risk_per_trade_pct": 1.0,  # "1" meaning 1% — must be 0.01
        },
    )
    assert is_error is True
    assert "fraction" in out


def test_build_options_ticket_loader_failure_is_error_not_crash(monkeypatch):
    import tradinglib.assistant.planner as planner

    def boom(*a, **k):
        raise ConnectionError("yfinance 429")

    monkeypatch.setattr(planner, "load_daily", boom)

    out, is_error = dispatch(
        "build_options_ticket",
        {
            "ticker": "RIVN",
            "stance": "long",
            "entry": 100.0,
            "stop": 96.0,
            "target": 108.0,
            "account_size": 100_000.0,
            "risk_per_trade_pct": 0.01,
        },
    )
    assert is_error is True
    assert "RIVN" in out
