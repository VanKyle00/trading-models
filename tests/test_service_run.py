# tests/test_service_run.py
import json
from datetime import date

import pytest

from tradinglib.service import BacktestRequest, BacktestRun, run, run_to_dict

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
