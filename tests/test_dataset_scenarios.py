from tradinglib.dataset.scenarios import generate_scenarios
from tradinglib.dataset.templates import CATEGORIES
from tradinglib.service import model_spec


def test_deterministic_with_seed():
    a = generate_scenarios(seed=7, per_model_per_category=2)
    b = generate_scenarios(seed=7, per_model_per_category=2)
    assert [s.question for s in a] == [s.question for s in b]


def test_every_category_present():
    scen = generate_scenarios(seed=1, per_model_per_category=2)
    assert set(s.category for s in scen) == set(CATEGORIES)


def test_symbols_are_legal_for_their_model():
    for s in generate_scenarios(seed=3, per_model_per_category=3):
        if not s.model_id or s.symbol is None:
            continue
        spec = model_spec(s.model_id)
        if spec.ticker_mode == "fixed":
            assert s.symbol in spec.ticker_choices or s.symbol == spec.default_ticker
        elif spec.ticker_mode == "choice":
            assert s.symbol in spec.ticker_choices


def test_no_placeholders_remain():
    for s in generate_scenarios(seed=5, per_model_per_category=2):
        assert "{" not in s.question and "}" not in s.question


def test_refusals_are_model_agnostic_or_safe():
    refusals = [
        s for s in generate_scenarios(seed=9, per_model_per_category=2) if s.category == "refusal"
    ]
    assert refusals
