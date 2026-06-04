# tradinglib/assistant/budget.py
"""Per-session spend caps + per-IP rate limiting.

The agent runs on the host's API budget, so a public visitor must not be able to
drive unbounded cost. Budget is per chat session; RateLimiter is process-global.
"""

from __future__ import annotations

from collections import defaultdict


class BudgetExceeded(RuntimeError):  # noqa: N818
    """Raised when a chat session hits its token or tool-call ceiling."""


class Budget:
    def __init__(self, max_tool_calls: int = 8, max_tokens: int = 60_000) -> None:
        self.max_tool_calls = max_tool_calls
        self.max_tokens = max_tokens
        self.tool_calls_used = 0
        self.tokens_used = 0

    def charge_tokens(self, n: int) -> None:
        if self.tokens_used + n > self.max_tokens:
            raise BudgetExceeded(
                f"token budget exhausted ({self.tokens_used + n} > {self.max_tokens})"
            )
        self.tokens_used += n

    def charge_tool_call(self) -> None:
        if self.tool_calls_used + 1 > self.max_tool_calls:
            raise BudgetExceeded(
                f"tool-call budget exhausted ({self.tool_calls_used + 1} > {self.max_tool_calls})"
            )
        self.tool_calls_used += 1


class RateLimiter:
    """Naive fixed-count in-memory limiter (per process). Good enough for a
    single-VPS demo; swap for Redis if horizontally scaled."""

    def __init__(self, max_per_window: int = 20) -> None:
        self.max_per_window = max_per_window
        self._counts: dict[str, int] = defaultdict(int)

    def allow(self, key: str) -> bool:
        if self._counts[key] >= self.max_per_window:
            return False
        self._counts[key] += 1
        return True

    def reset(self) -> None:
        self._counts.clear()
