# tests/test_service_run.py
import json
from datetime import date

import pandas as pd
import pytest

from tradinglib.service import BacktestRequest, BacktestRun, list_specs, run, run_to_dict

CLASSICAL = "models/classical/01-sma-crossover-spy"

# Top-level keys run_to_dict guarantees forever. New keys may be ADDED; these
# may never be removed or repurposed (additive contract — see spec).
BASELINE_KEYS = {"model_id", "symbol", "params", "metrics", "config", "series", "trades"}


@pytest.fixture(scope="module")
def classical_run():
    req = BacktestRequest(
        model_id=CLASSICAL,
        symbol="SPY",
        start=date(2022, 1, 1),
        end=date(2023, 1, 1),
        params={"fast": 20, "slow": 50},
    )
    return run(req)


def test_run_returns_backtest_run(classical_run):
    assert isinstance(classical_run, BacktestRun)
    assert classical_run.symbol == "SPY"
    assert "sharpe" in classical_run.result.metrics


def test_run_to_dict_is_json_serializable(classical_run):
    d = run_to_dict(classical_run)
    json.dumps(d)  # must not raise


def test_run_to_dict_has_baseline_keys(classical_run):
    d = run_to_dict(classical_run)
    assert set(d) >= BASELINE_KEYS  # superset — additive contract
    assert "equity" in d["series"]
    assert isinstance(d["trades"], list)


def test_run_to_dict_downsamples_long_series(classical_run):
    d = run_to_dict(classical_run)
    assert len(d["series"]["equity"]["values"]) <= 1000


def test_invalid_request_raises_before_dispatch():
    from tradinglib.service import RequestError

    bad = BacktestRequest(
        model_id=CLASSICAL,
        symbol="SPY",
        start=date(2023, 1, 1),
        end=date(2022, 1, 1),  # reversed
    )
    with pytest.raises(RequestError):
        run(bad)


def test_series_downsamples_and_keeps_last_point():
    from tradinglib.service.run import _MAX_POINTS, _series

    idx = pd.date_range("2000-01-01", periods=5000, freq="D")
    s = pd.Series(range(5000), index=idx, dtype=float)
    out = _series(s)
    assert len(out["values"]) <= _MAX_POINTS
    assert out["values"][-1] == 4999.0  # terminal bar preserved
    assert out["index"][-1] == idx[-1].isoformat()


def test_jsonify_coerces_numpy_scalars():
    import numpy as np

    from tradinglib.service.run import _jsonify

    out = _jsonify({"a": np.int64(50), "b": np.float64(1.5), "c": "x", "d": [np.int64(7)]})
    assert out == {"a": 50, "b": 1.5, "c": "x", "d": [7]}
    json.dumps(out)  # must not raise


def _smoke_window(spec):
    # Each locked model only has data in a specific window; use generous ranges.
    if spec.family == "microstructure":
        return date(2024, 8, 4), date(2024, 8, 6)
    if spec.family == "alt-data":
        return date(2022, 1, 1), date(2023, 1, 1)
    return date(2023, 1, 1), date(2024, 1, 1)


@pytest.mark.parametrize("spec", list_specs(), ids=lambda s: s.family)
def test_every_model_runs_and_serializes(spec):
    start, end = _smoke_window(spec)
    req = BacktestRequest(model_id=spec.id, start=start, end=end)  # symbol=None → default
    try:
        result = run_to_dict(run(req))
    except (FileNotFoundError, ConnectionError, OSError) as exc:
        pytest.skip(f"{spec.family}: data unavailable in this environment ({exc})")
    assert set(result) >= BASELINE_KEYS
    json.dumps(result)


def test_run_to_dict_baseline_value_types(classical_run):
    # Additive contract: baseline keys must keep their value TYPES, not just exist.
    d = run_to_dict(classical_run)
    assert isinstance(d["model_id"], str)
    assert isinstance(d["symbol"], str)
    assert isinstance(d["params"], dict)
    assert isinstance(d["metrics"], dict)
    assert isinstance(d["config"], dict)
    assert isinstance(d["series"], dict)
    assert isinstance(d["trades"], list)


def test_backtest_request_optional_fields_construct_with_defaults():
    # Additive contract: every non-core field must stay optional.
    req = BacktestRequest(model_id=CLASSICAL, start=date(2022, 1, 1), end=date(2022, 6, 1))
    assert req.symbol is None
    assert req.fee_bps is None and req.slippage_bps is None
    assert req.initial_capital is None and req.size_mult is None
    assert req.params == {}
