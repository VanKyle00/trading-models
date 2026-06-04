# tests/test_webapp_forms.py
from datetime import date

import pytest

from tradinglib.service import RequestError, model_spec
from webapp.forms import request_from_payload

CLASSICAL = "models/classical/01-sma-crossover-spy"


def test_builds_request_with_typed_params():
    payload = {
        "model_id": CLASSICAL,
        "symbol": "spy",
        "start": "2022-01-01",
        "end": "2023-01-01",
        "fast": "30",
        "slow": "100",
    }
    req = request_from_payload(payload, model_spec(CLASSICAL))
    assert req.start == date(2022, 1, 1)
    assert req.end == date(2023, 1, 1)
    assert req.symbol == "spy"  # raw; service.run normalizes
    assert req.params == {"fast": 30, "slow": 100}  # coerced to int per spec


def test_float_param_coerced():
    options = "models/options/01-delta-hedged-long-option-spy"
    payload = {
        "model_id": options,
        "start": "2023-01-01",
        "end": "2023-06-01",
        "implied_vol": "0.25",
        "tenor_days": "30",
        "n_paths": "2000",
    }
    req = request_from_payload(payload, model_spec(options))
    assert req.params["implied_vol"] == 0.25
    assert isinstance(req.params["implied_vol"], float)
    assert req.params["tenor_days"] == 30


def test_blank_optional_costs_omitted():
    payload = {
        "model_id": CLASSICAL,
        "start": "2022-01-01",
        "end": "2023-01-01",
        "fee_bps": "",
        "size_mult": "",
    }
    req = request_from_payload(payload, model_spec(CLASSICAL))
    assert req.fee_bps is None and req.size_mult is None


def test_bad_date_raises_request_error():
    payload = {"model_id": CLASSICAL, "start": "not-a-date", "end": "2023-01-01"}
    with pytest.raises(RequestError, match="date"):
        request_from_payload(payload, model_spec(CLASSICAL))


def test_blank_symbol_becomes_none():
    payload = {"model_id": CLASSICAL, "symbol": "", "start": "2022-01-01", "end": "2023-01-01"}
    req = request_from_payload(payload, model_spec(CLASSICAL))
    assert req.symbol is None


def test_list_symbol_is_coerced_to_str_not_crashing_validate():
    from tradinglib.service import validate

    payload = {"model_id": CLASSICAL, "symbol": ["SPY"], "start": "2022-01-01", "end": "2023-01-01"}
    req = request_from_payload(payload, model_spec(CLASSICAL))
    assert req.symbol == "SPY"
    assert isinstance(req.symbol, str)
    validate(req, model_spec(CLASSICAL))  # must not raise AttributeError
