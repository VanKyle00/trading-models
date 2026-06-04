# tests/test_assistant_budget.py
import pytest

from tradinglib.assistant.budget import Budget, BudgetExceeded, RateLimiter


def test_budget_allows_within_caps():
    b = Budget(max_tool_calls=3, max_tokens=1000)
    b.charge_tokens(400)
    b.charge_tool_call()
    assert b.tool_calls_used == 1
    assert b.tokens_used == 400


def test_budget_blocks_on_token_cap():
    b = Budget(max_tool_calls=10, max_tokens=500)
    b.charge_tokens(400)
    with pytest.raises(BudgetExceeded, match="token"):
        b.charge_tokens(200)


def test_budget_blocks_on_tool_call_cap():
    b = Budget(max_tool_calls=2, max_tokens=10_000)
    b.charge_tool_call()
    b.charge_tool_call()
    with pytest.raises(BudgetExceeded, match="tool"):
        b.charge_tool_call()


def test_rate_limiter_per_key():
    rl = RateLimiter(max_per_window=2)
    assert rl.allow("ip-a") is True
    assert rl.allow("ip-a") is True
    assert rl.allow("ip-a") is False
    assert rl.allow("ip-b") is True
