"""LLM-judge for answer quality: candidate vs the gold (Claude-teacher) answer.

ClaudeJudge presents the two answers in a deterministic but content-derived order
(to control position bias without nondeterminism), asks which better answers the
question, and maps the verdict to the candidate's perspective. StubJudge scripts
verdicts for tests. anthropic is imported lazily so this module loads without a key."""

from __future__ import annotations

import hashlib
import re

Verdict = str  # "win" | "tie" | "loss"

_JUDGE_SYSTEM = (
    "You are grading two answers to a quantitative-research question about backtest "
    "results. Pick the one that is more accurate, better grounded in the numbers, and "
    "more useful. If they are equivalent, say TIE. Reply with exactly one line: "
    "'Answer: A', 'Answer: B', or 'Answer: TIE'."
)


def _parse_verdict(text: str, candidate_label: str) -> Verdict:
    m = re.search(r"\b(A|B|TIE)\b", text.upper())
    if not m:
        return "tie"
    pick = m.group(1)
    if pick == "TIE":
        return "tie"
    return "win" if pick == candidate_label else "loss"


def _candidate_is_a(question: str, candidate: str) -> bool:
    """Pick the candidate's A/B slot from a STABLE content hash, so the same
    (question, candidate) always lands in the same slot across runs/processes
    (builtin hash() is per-process salted and would not be reproducible)."""
    digest = hashlib.sha256(f"{question}\x00{candidate}".encode()).digest()
    return digest[0] % 2 == 0


def win_rate(verdicts: list[Verdict]) -> float:
    if not verdicts:
        return 0.0
    wins = sum(v == "win" for v in verdicts)
    ties = sum(v == "tie" for v in verdicts)
    return (wins + 0.5 * ties) / len(verdicts)


class StubJudge:
    def __init__(self, verdicts: list[Verdict]) -> None:
        self._verdicts = list(verdicts)

    def compare(self, question: str, gold: str, candidate: str) -> Verdict:
        if not self._verdicts:
            raise AssertionError("StubJudge ran out of scripted verdicts")
        return self._verdicts.pop(0)


class ClaudeJudge:
    def __init__(self, model: str = "claude-haiku-4-5", max_tokens: int = 16) -> None:
        import anthropic  # lazy: stub path and CI need no key

        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def compare(self, question: str, gold: str, candidate: str) -> Verdict:
        # Stable content-derived order: candidate is A when the digest is even.
        cand_is_a = _candidate_is_a(question, candidate)
        a, b = (candidate, gold) if cand_is_a else (gold, candidate)
        label = "A" if cand_is_a else "B"
        prompt = (
            f"Question:\n{question}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}\n\n"
            "Which answer is better? Reply 'Answer: A', 'Answer: B', or 'Answer: TIE'."
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(blk.text for blk in resp.content if blk.type == "text")
        return _parse_verdict(text, label)
