"""Bridge LLMProvider that runs our fine-tuned model on a remote Modal GPU.

Selected only inside the Modal deployment (when ``ASSISTANT_LOCAL_BACKEND=modal``,
set on the web container's image in ``deploy/modal_app.py``). WSL2/local GPU and
tests use the in-process ``LocalAdapterProvider`` instead, so the ``modal`` import
here stays lazy and the rest of the app stays modal-free.

The GPU class (``LocalModel``) lives in the same Modal app; we look it up by name
and call ``complete`` over the wire. The neutral conversation/response types are
plain frozen dataclasses, so Modal (cloudpickle) ships them across without any
manual serialization.
"""

from __future__ import annotations

from typing import Any

from tradinglib.assistant.types import AssistantTurn, Message

_MODAL_APP = "trading-models-workbench"
_MODAL_CLS = "LocalModel"


class ModalRemoteProvider:
    """Implements the LLMProvider protocol by calling the Modal GPU container."""

    def __init__(self) -> None:
        import modal

        # Resolves at call time; the GPU container cold-starts on the first
        # .remote() and scales back to zero after its idle window.
        self._model = modal.Cls.from_name(_MODAL_APP, _MODAL_CLS)()

    def complete(
        self, system: str, conversation: list[Message], tools: list[dict[str, Any]]
    ) -> AssistantTurn:
        return self._model.complete.remote(system, conversation, tools)
