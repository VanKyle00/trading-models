# tests/test_assistant_tools.py
import json

from tradinglib.assistant.tools import TOOL_SPECS, dispatch


def test_tool_specs_are_the_three_service_tools():
    names = {t["name"] for t in TOOL_SPECS}
    assert names == {"list_models", "get_model_spec", "run_backtest"}
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
