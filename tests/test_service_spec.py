# tests/test_service_spec.py
import pytest

from tradinglib.service import ModelSpec, ParamSpec, list_specs, model_spec


def test_lists_all_five_models():
    specs = list_specs()
    assert len(specs) == 5
    assert all(isinstance(s, ModelSpec) for s in specs)


def test_classical_spec_shape():
    spec = next(s for s in list_specs() if s.family == "classical")
    assert spec.ticker_mode == "free"
    assert spec.default_ticker == "SPY"
    assert spec.supports_costs is True
    assert spec.supports_sizing is True
    names = [p.name for p in spec.params]
    assert names == ["fast", "slow"]
    fast = spec.params[0]
    assert isinstance(fast, ParamSpec)
    assert (fast.type, fast.default, fast.min, fast.max) == ("int", 50, 5, 150)


def test_locked_model_modes_and_capabilities():
    ml = next(s for s in list_specs() if s.family == "ml")
    assert ml.ticker_mode == "fixed"
    assert ml.default_ticker == "SPY"
    assert ml.params == ()
    micro = next(s for s in list_specs() if s.family == "microstructure")
    assert micro.supports_costs is False
    assert micro.supports_sizing is False


def test_model_spec_by_id_roundtrips():
    spec = list_specs()[0]
    assert model_spec(spec.id) is not None
    assert model_spec(spec.id).id == spec.id


def test_model_spec_unknown_id_raises():
    with pytest.raises(KeyError):
        model_spec("models/nope/does-not-exist")


def test_every_model_declares_params_key():
    for spec in list_specs():
        assert isinstance(spec.params, tuple)
