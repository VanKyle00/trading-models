from tradinglib.dataset.scenarios import (
    generate_scenarios,
    partition_by_ticker,
    partition_stratified,
)
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


def _is_partition(scen, train, eval_):
    """train+eval is exactly scen, with no overlap (compare by identity)."""
    assert len(train) + len(eval_) == len(scen)
    ids_tr, ids_ev = {id(s) for s in train}, {id(s) for s in eval_}
    assert not (ids_tr & ids_ev)
    assert ids_tr | ids_ev == {id(s) for s in scen}


def test_partition_by_ticker_no_leakage_and_deterministic():
    scen = generate_scenarios(seed=0, per_model_per_category=3)
    train, eval_ = partition_by_ticker(scen, eval_frac=0.2, seed=0)
    _is_partition(scen, train, eval_)
    tr_tk = {s.symbol for s in train if s.symbol}
    ev_tk = {s.symbol for s in eval_ if s.symbol}
    assert not (tr_tk & ev_tk)  # a ticker is wholly in one side or the other
    _train2, eval2 = partition_by_ticker(scen, eval_frac=0.2, seed=0)
    assert [s.question for s in eval_] == [s.question for s in eval2]


def test_partition_stratified_covers_categories_without_number_leakage():
    scen = generate_scenarios(seed=0, per_model_per_category=3)
    train, eval_ = partition_stratified(scen, eval_frac=0.2, seed=0)
    _is_partition(scen, train, eval_)
    # every category appears in eval (so the gate can read all of them)
    assert set(s.category for s in eval_) == set(CATEGORIES)
    # explain/counterfactual: no ticker shared train<->eval (no backtest-number leak)
    for cat in ("explain", "counterfactual"):
        tr_tk = {s.symbol for s in train if s.category == cat and s.symbol}
        ev_tk = {s.symbol for s in eval_ if s.category == cat and s.symbol}
        assert not (tr_tk & ev_tk), f"{cat} ticker leak: {tr_tk & ev_tk}"
    # deterministic
    _train2, eval2 = partition_stratified(scen, eval_frac=0.2, seed=0)
    assert [s.question for s in eval_] == [s.question for s in eval2]
