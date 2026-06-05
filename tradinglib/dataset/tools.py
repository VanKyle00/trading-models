"""Dataset-time tool set = the three runtime tools + a `search_docs` over the
RAG corpus, so methodology traces are grounded in retrieved doc chunks.

Runtime `tradinglib.assistant.tools` is unchanged; we delegate to its `dispatch`
for the shared tools and add `search_docs` here. Returns the same
(content_json, is_error) contract the agent loop expects."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from tradinglib.assistant.tools import TOOL_SPECS
from tradinglib.assistant.tools import dispatch as runtime_dispatch
from tradinglib.dataset.corpus import Corpus

_SEARCH_DOCS_SPEC = {
    "name": "search_docs",
    "description": (
        "Search the project's methodology/docs for grounding. Use for questions "
        "about how backtests work, metrics, data sources, or model design."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
        "required": ["query"],
    },
}

DispatchFn = Callable[[str, dict[str, Any]], tuple[str, bool]]


def make_dataset_tools(corpus: Corpus) -> tuple[list[dict[str, Any]], DispatchFn]:
    specs = [*TOOL_SPECS, _SEARCH_DOCS_SPEC]

    def dispatch(name: str, args: dict[str, Any]) -> tuple[str, bool]:
        if name == "search_docs":
            try:
                hits = corpus.search(str(args.get("query", "")), int(args.get("k", 3)))
                return json.dumps({"chunks": hits}), False
            except Exception as exc:  # tool boundary must never raise
                return f"search_docs failed: {type(exc).__name__}: {exc}", True
        return runtime_dispatch(name, args)

    return specs, dispatch
