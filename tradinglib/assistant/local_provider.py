"""LocalAdapterProvider: serve base Qwen2.5-7B + our LoRA adapter locally (WSL2/GPU).

Implements LLMProvider so the eval harness (and, later, anything else) can point at
our own trained model with no agent-loop changes. Heavy deps (torch/transformers/peft/
bitsandbytes, the [train] optional group) are imported lazily inside the constructor and
complete(), so this module imports in the GPU-free CI env. The Qwen tool-call PARSER is a
pure module-level function, unit-tested without a model load -- it is the riskiest piece
and is also operator-verified at smoke time.

This file is exercised for real only in WSL2 after training; CI covers the parser + the
lazy-import guarantee."""

from __future__ import annotations

import json
import re
from typing import Any

from tradinglib.assistant.types import AssistantTurn, ToolCall, Usage

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_qwen_tool_calls(text: str) -> tuple[list[ToolCall], str, str]:
    """Parse Qwen-style <tool_call>{json}</tool_call> blocks.

    Returns (calls, stop_reason, prose). stop_reason is "tool_use" if any valid call
    was parsed, else "end_turn". prose is the text with tool-call blocks stripped.
    Malformed JSON blocks are skipped (treated as no call)."""
    calls: list[ToolCall] = []
    for i, m in enumerate(_TOOL_CALL_RE.finditer(text)):
        try:
            obj = json.loads(m.group(1))
            name = obj["name"]
            args = obj.get("arguments", {})
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append(ToolCall(id=f"call_{i}", name=name, input=args))
    prose = _TOOL_CALL_RE.sub("", text)
    stop_reason = "tool_use" if calls else "end_turn"
    return calls, stop_reason, prose


def _to_chat_messages(system: str, conversation: list) -> list[dict[str, Any]]:
    """Render the neutral conversation into OpenAI/Qwen chat-template messages."""
    from tradinglib.assistant.types import AssistantMsg, ToolResultMsg, UserMsg

    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in conversation:
        if isinstance(m, UserMsg):
            out.append({"role": "user", "content": m.text})
        elif isinstance(m, AssistantMsg):
            entry: dict[str, Any] = {"role": "assistant", "content": m.turn.text}
            if m.turn.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.input)},
                    }
                    for tc in m.turn.tool_calls
                ]
            out.append(entry)
        elif isinstance(m, ToolResultMsg):
            for r in m.results:
                out.append({"role": "tool", "tool_call_id": r.tool_call_id, "content": r.content})
    return out


class LocalAdapterProvider:
    """Load base + LoRA adapter and generate. GPU/operator-run; not used in CI."""

    def __init__(
        self,
        adapter_path: str,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens: int = 512,
        load_in_4bit: bool = True,
    ) -> None:
        import torch  # noqa: F401  lazy: GPU-only
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Prefer the tokenizer saved with the adapter: it carries the exact chat
        # template + special tokens used at train time, so the eval prompt format
        # matches training. Fall back to the base model's tokenizer.
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        except (OSError, ValueError):
            self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        quant = (
            BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16"
            )
            if load_in_4bit
            else None
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=quant, device_map="auto"
        )
        self._model = PeftModel.from_pretrained(base, adapter_path)
        self._model.eval()
        self._max_new_tokens = max_new_tokens

    def complete(
        self, system: str, conversation: list, tools: list[dict[str, Any]]
    ) -> AssistantTurn:
        import torch

        messages = _to_chat_messages(system, conversation)
        prompt = self._tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens, do_sample=False
            )
        gen = self._tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        calls, stop_reason, prose = parse_qwen_tool_calls(gen)
        return AssistantTurn(
            text=prose.strip(),
            tool_calls=tuple(calls),
            stop_reason=stop_reason,
            usage=Usage(input_tokens=prompt_len, output_tokens=out[0].shape[0] - prompt_len),
        )
