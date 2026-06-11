# tests/test_assistant_agent.py
from tradinglib.assistant.agent import run_chat
from tradinglib.assistant.budget import Budget
from tradinglib.assistant.provider import StubProvider
from tradinglib.assistant.types import AssistantMsg, AssistantTurn, ToolCall, Usage, UserMsg


def _events(gen):
    return list(gen)


def test_plain_answer_no_tools():
    provider = StubProvider(
        [
            AssistantTurn(
                text="The SMA model is a baseline.",
                tool_calls=(),
                stop_reason="end_turn",
                usage=Usage(10, 8),
            ),
        ]
    )
    events = _events(run_chat("what is the sma model?", provider, Budget()))
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "final"
    assert "baseline" in events[-1]["text"]
    assert kinds.count("text") == 0  # terminal answer goes only in 'final', not duplicated
    assert kinds.count("final") == 1


def test_runs_a_tool_then_answers():
    provider = StubProvider(
        [
            AssistantTurn(
                text="Looking it up.",
                tool_calls=(ToolCall("t1", "list_models", {}),),
                stop_reason="tool_use",
                usage=Usage(5, 5),
            ),
            AssistantTurn(
                text="There are five models.",
                tool_calls=(),
                stop_reason="end_turn",
                usage=Usage(5, 5),
            ),
        ]
    )
    events = _events(run_chat("how many models?", provider, Budget()))
    kinds = [e["type"] for e in events]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert kinds[-1] == "final"
    assert "five" in events[-1]["text"]


def test_context_is_folded_into_opening_message():
    provider = StubProvider(
        [AssistantTurn(text="ok", tool_calls=(), stop_reason="end_turn", usage=Usage(3, 3))]
    )
    _events(
        run_chat(
            "explain the worst month",
            provider,
            Budget(),
            context="Backtest: Momentum · SPY\nMetrics: Sharpe=1.42",
        )
    )
    opening = provider.calls[0][0].text  # first UserMsg of the first turn
    assert "Sharpe=1.42" in opening
    assert "explain the worst month" in opening


def test_no_context_leaves_message_verbatim():
    provider = StubProvider(
        [AssistantTurn(text="ok", tool_calls=(), stop_reason="end_turn", usage=Usage(3, 3))]
    )
    _events(run_chat("hello", provider, Budget()))
    assert provider.calls[0][0].text == "hello"


def test_budget_exhaustion_ends_gracefully():
    looping = [
        AssistantTurn(
            text="",
            tool_calls=(ToolCall(f"t{i}", "list_models", {}),),
            stop_reason="tool_use",
            usage=Usage(1, 1),
        )
        for i in range(5)
    ]
    looping_provider = StubProvider(looping)
    events = _events(
        run_chat("loop", looping_provider, Budget(max_tool_calls=1, max_tokens=10_000))
    )
    assert events[-1]["type"] == "final"
    assert "limit" in events[-1]["text"].lower()
    assert looping_provider.calls


def _final(text="ok"):
    return AssistantTurn(text=text, tool_calls=(), stop_reason="end_turn", usage=Usage(3, 3))


def test_history_seeds_conversation_in_order():
    provider = StubProvider([_final()])
    _events(
        run_chat(
            "and the target?",
            provider,
            Budget(),
            history=[("user", "I'm bullish on RIVN"), ("assistant", "Proposed: entry 14.2")],
        )
    )
    msgs = provider.calls[0]
    assert isinstance(msgs[0], UserMsg) and msgs[0].text == "I'm bullish on RIVN"
    assert isinstance(msgs[1], AssistantMsg) and msgs[1].turn.text == "Proposed: entry 14.2"
    assert isinstance(msgs[2], UserMsg) and msgs[2].text == "and the target?"


def test_history_merges_consecutive_same_role_messages():
    provider = StubProvider([_final()])
    _events(
        run_chat(
            "next",
            provider,
            Budget(),
            history=[("user", "a"), ("user", "b"), ("assistant", "c"), ("user", "d")],
        )
    )
    msgs = provider.calls[0]
    assert len(msgs) == 3
    assert msgs[0].text == "a\n\nb"
    assert msgs[1].turn.text == "c"
    assert msgs[2].text == "d\n\nnext"  # trailing user history merges into the opening


def test_history_drops_leading_assistant_and_empty_texts():
    provider = StubProvider([_final()])
    _events(
        run_chat(
            "q2",
            provider,
            Budget(),
            history=[
                ("assistant", "orphan greeting"),
                ("user", ""),
                ("user", "q1"),
                ("assistant", "a1"),
            ],
        )
    )
    msgs = provider.calls[0]
    assert isinstance(msgs[0], UserMsg) and msgs[0].text == "q1"
    assert isinstance(msgs[1], AssistantMsg)
    assert msgs[2].text == "q2"


def test_history_with_context_folds_context_into_opening_only():
    provider = StubProvider([_final()])
    _events(
        run_chat(
            "explain",
            provider,
            Budget(),
            context="Backtest: SMA · SPY",
            history=[("user", "hello"), ("assistant", "hi")],
        )
    )
    msgs = provider.calls[0]
    assert msgs[0].text == "hello"  # history left verbatim
    assert "Backtest: SMA · SPY" in msgs[2].text and "explain" in msgs[2].text


def test_tool_result_events_carry_output():
    provider = StubProvider(
        [
            AssistantTurn(
                text="",
                tool_calls=(ToolCall("t1", "list_models", {}),),
                stop_reason="tool_use",
                usage=Usage(5, 5),
            ),
            _final("done"),
        ]
    )
    events = _events(run_chat("list them", provider, Budget()))
    tr = next(e for e in events if e["type"] == "tool_result")
    assert "output" in tr
    assert "models" in tr["output"]  # list_models returns a JSON string with a models key


def test_settings_folded_into_opening_message():
    provider = StubProvider([_final()])
    _events(
        run_chat(
            "I'm bullish on RIVN",
            provider,
            Budget(),
            settings=(
                "Planner sizing (set on the page): account size $50,000; risk per trade 2% (0.02)."
            ),
        )
    )
    opening = provider.calls[0][0].text
    assert "I'm bullish on RIVN" in opening
    assert "$50,000" in opening
    assert "do not ask the user for account size or risk" in opening


def test_settings_and_context_both_fold_into_opening():
    provider = StubProvider([_final()])
    _events(
        run_chat(
            "explain",
            provider,
            Budget(),
            context="Backtest: SMA · SPY",
            settings=(
                "Planner sizing (set on the page): account size $100,000; risk per trade 1% (0.01)."
            ),
        )
    )
    opening = provider.calls[0][0].text
    assert "SMA" in opening and "$100,000" in opening and "explain" in opening


def test_system_prompt_handles_page_sizing_settings():
    from tradinglib.assistant.provider import SYSTEM_PROMPT

    assert "planner sizing settings" in SYSTEM_PROMPT.lower()
    assert "confirm only the scenario" in SYSTEM_PROMPT.lower()
