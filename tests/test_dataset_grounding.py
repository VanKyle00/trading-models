from tradinglib.dataset.grounding import extract_numbers, is_grounded


def test_extract_plain_and_percent_and_signed():
    nums = extract_numbers("Sharpe 0.96, drawdown -12%, equity $10,000.")
    assert 0.96 in nums
    assert -0.12 in nums  # -12% normalized to -0.12
    assert 10000.0 in nums


def test_grounded_answer_passes():
    tool_outputs = ['{"metrics": {"sharpe": 0.9612, "max_drawdown": -0.121}}']
    ok, missing = is_grounded("Sharpe is about 0.96 with a -12% drawdown.", tool_outputs)
    assert ok and missing == []


def test_hallucinated_number_is_caught():
    tool_outputs = ['{"metrics": {"sharpe": 0.96}}']
    ok, missing = is_grounded("Sharpe is 2.31, excellent.", tool_outputs)
    assert not ok and 2.31 in missing


def test_small_integers_are_ignored():
    # bare 1/2/3 etc. are prose ("the 3 models"), not claims to verify
    ok, missing = is_grounded("There are 5 models and 2 asset classes.", [])
    assert ok and missing == []


def test_index_name_numbers_are_not_metric_claims():
    # "S&P 500" / "Russell 2000" are proper-noun names, not numbers to verify
    assert extract_numbers("The S&P 500 and Russell 2000 indices") == set()
    ok, missing = is_grounded("Tracks the S&P 500; Sharpe was 1.2.", ['{"sharpe": 1.2}'])
    assert ok and missing == []
