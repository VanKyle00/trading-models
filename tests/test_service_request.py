# tests/test_service_request.py
from datetime import date

import pytest

from tradinglib.service import BacktestRequest, RequestError, model_spec, validate

CLASSICAL = "models/classical/01-sma-crossover-spy"
MICRO = "models/microstructure/01-order-flow-btc"


def _req(**kw):
    base = dict(model_id=CLASSICAL, symbol="SPY", start=date(2022, 1, 1), end=date(2022, 6, 1))
    base.update(kw)
    return BacktestRequest(**base)


def test_valid_request_passes():
    validate(_req(params={"fast": 30, "slow": 100}), model_spec(CLASSICAL))  # no raise


def test_bad_date_order_rejected():
    with pytest.raises(RequestError, match="start"):
        validate(_req(start=date(2022, 6, 1), end=date(2022, 1, 1)), model_spec(CLASSICAL))


def test_unknown_param_rejected():
    with pytest.raises(RequestError, match="unknown param"):
        validate(_req(params={"nope": 1}), model_spec(CLASSICAL))


def test_param_out_of_range_rejected():
    with pytest.raises(RequestError, match="range"):
        validate(_req(params={"fast": 999}), model_spec(CLASSICAL))


def test_param_wrong_type_rejected():
    with pytest.raises(RequestError, match="type"):
        validate(_req(params={"fast": 1.5}), model_spec(CLASSICAL))


def test_fixed_ticker_must_match():
    ml = "models/ml/01-gbm-next-day-return-spy"
    with pytest.raises(RequestError, match="locked"):
        validate(
            BacktestRequest(
                model_id=ml, symbol="QQQ", start=date(2022, 1, 1), end=date(2022, 6, 1)
            ),
            model_spec(ml),
        )


def test_cost_override_rejected_on_incapable_model():
    with pytest.raises(RequestError, match="cost"):
        validate(
            BacktestRequest(
                model_id=MICRO,
                symbol="BTCUSDT",
                start=date(2024, 8, 4),
                end=date(2024, 8, 6),
                fee_bps=5.0,
            ),
            model_spec(MICRO),
        )


def test_none_symbol_uses_default_ok():
    validate(_req(symbol=None), model_spec(CLASSICAL))  # no raise; run() fills the default


def test_bool_rejected_for_float_param():
    options = "models/options/01-delta-hedged-long-option-spy"
    with pytest.raises(RequestError, match="type"):
        validate(
            BacktestRequest(
                model_id=options,
                symbol="SPY",
                start=date(2023, 1, 1),
                end=date(2023, 6, 1),
                params={"implied_vol": True},
            ),
            model_spec(options),
        )


def test_initial_capital_override_rejected_on_incapable_model():
    with pytest.raises(RequestError, match="sizing"):
        validate(
            BacktestRequest(
                model_id=MICRO,
                symbol="BTCUSDT",
                start=date(2024, 8, 4),
                end=date(2024, 8, 6),
                initial_capital=50_000.0,
            ),
            model_spec(MICRO),
        )


def test_sizing_and_capital_override_ok_on_capable_model():
    validate(_req(size_mult=2.0, initial_capital=50_000.0), model_spec(CLASSICAL))  # no raise


def test_negative_fee_rejected():
    with pytest.raises(RequestError, match="fee_bps"):
        validate(_req(fee_bps=-1.0), model_spec(CLASSICAL))


def test_nonpositive_initial_capital_rejected():
    with pytest.raises(RequestError, match="initial_capital"):
        validate(_req(initial_capital=0.0), model_spec(CLASSICAL))


def test_size_mult_out_of_bounds_rejected():
    with pytest.raises(RequestError, match="size_mult"):
        validate(_req(size_mult=0.0), model_spec(CLASSICAL))
    with pytest.raises(RequestError, match="size_mult"):
        validate(_req(size_mult=100.0), model_spec(CLASSICAL))


def test_reasonable_overrides_pass():
    validate(
        _req(fee_bps=2.0, slippage_bps=1.0, initial_capital=250_000.0, size_mult=2.0),
        model_spec(CLASSICAL),
    )
