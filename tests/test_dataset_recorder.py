from tradinglib.assistant.budget import Budget
from tradinglib.assistant.provider import StubProvider
from tradinglib.assistant.types import AssistantTurn, ToolCall, Usage
from tradinglib.dataset.corpus import Corpus
from tradinglib.dataset.recorder import record_trace
from tradinglib.dataset.scenarios import Scenario
from tradinglib.dataset.tools import make_dataset_tools


def _turn(text="", calls=(), stop="end_turn"):
    return AssistantTurn(
        text=text,
        tool_calls=tuple(calls),
        stop_reason=stop,
        usage=Usage(input_tokens=5, output_tokens=5),
    )


def _scn(category="methodology"):
    return Scenario(
        category=category,
        question="How are fills modelled?",
        model_id="",
        symbol=None,
        start="2020-01-01",
        end="2023-12-31",
    )


def test_records_final_answer_and_tool_outputs():
    specs, dispatch = make_dataset_tools(Corpus.from_chunks(["Fills use next-open execution."]))
    provider = StubProvider(
        [
            _turn(
                calls=[ToolCall(id="t1", name="search_docs", input={"query": "fills"})],
                stop="tool_use",
            ),
            _turn(text="Fills use next-open execution."),
        ]
    )
    trace = record_trace(_scn(), provider, Budget(), specs, dispatch)
    assert trace.ok
    assert trace.final_answer == "Fills use next-open execution."
    assert any("next-open" in o for o in trace.tool_outputs)
    assert len(trace.conversation) >= 3  # user, assistant(tool), tool_result


def test_marks_not_ok_when_no_final_text():
    specs, dispatch = make_dataset_tools(Corpus.from_chunks(["fills methodology"]))
    provider = StubProvider([_turn(text="")])  # empty final answer
    trace = record_trace(_scn(), provider, Budget(), specs, dispatch)
    assert not trace.ok
