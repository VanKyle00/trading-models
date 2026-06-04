# tradinglib/assistant/types.py
"""Provider-neutral conversation + response types.

The agent loop speaks ONLY these types. A concrete LLMProvider translates them
to/from its native wire format, so a different model backend can be swapped in
without touching agent.py or tools.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class AssistantTurn:
    """One model turn: free text plus any tool calls it wants to make."""

    text: str
    tool_calls: tuple[ToolCall, ...]
    stop_reason: str
    usage: Usage


@dataclass(frozen=True)
class UserMsg:
    text: str


@dataclass(frozen=True)
class AssistantMsg:
    turn: AssistantTurn


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ToolResultMsg:
    results: tuple[ToolResult, ...]


Message = UserMsg | AssistantMsg | ToolResultMsg
