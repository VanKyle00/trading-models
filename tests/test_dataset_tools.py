import json

from tradinglib.dataset.corpus import Corpus
from tradinglib.dataset.tools import make_dataset_tools


def _corpus():
    return Corpus.from_chunks(["Fills use next-open execution to avoid look-ahead."])


def test_includes_runtime_tools_plus_search_docs():
    specs, _ = make_dataset_tools(_corpus())
    names = {s["name"] for s in specs}
    assert {"list_models", "get_model_spec", "run_backtest", "search_docs"} <= names


def test_search_docs_returns_chunks_no_error():
    _, dispatch = make_dataset_tools(_corpus())
    content, is_error = dispatch("search_docs", {"query": "how are fills modelled?"})
    assert not is_error
    assert "next-open" in json.loads(content)["chunks"][0]


def test_delegates_known_tool_to_runtime_dispatch():
    _, dispatch = make_dataset_tools(_corpus())
    content, is_error = dispatch("list_models", {})
    assert not is_error and "models" in json.loads(content)


def test_unknown_tool_is_error_not_raise():
    _, dispatch = make_dataset_tools(_corpus())
    _content, is_error = dispatch("nope", {})
    assert is_error
