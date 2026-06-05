"""Orchestrate cases -> runner -> score -> judge, aggregate into a Scorecard.

ONE provider instance is reused across cases (real GPU/Claude providers are
stateless per call and must not be reloaded each case); a fresh Budget is created
per case. The Scorecard renders a markdown table and checks the ship-bar; the CLI
turns passes() into the process exit code."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from tradinglib.assistant.budget import Budget
from tradinglib.assistant_eval.config import DEFAULT_BAR, ShipBar
from tradinglib.assistant_eval.judge import win_rate
from tradinglib.assistant_eval.runner import run_candidate
from tradinglib.assistant_eval.score import grounded, tool_call_score


@dataclass
class Scorecard:
    n: int
    tool_call_accuracy: float
    grounded_fraction: float
    judge_winrate: float
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)

    def passes(self, bar: ShipBar = DEFAULT_BAR) -> bool:
        return (
            self.tool_call_accuracy >= bar.tool_call_accuracy
            and self.grounded_fraction >= bar.grounded_fraction
            and self.judge_winrate >= bar.judge_winrate
        )

    def render(self) -> str:
        lines = [
            "# Eval scorecard",
            "",
            f"- cases: {self.n}",
            f"- tool-call accuracy: {self.tool_call_accuracy:.3f}",
            f"- grounded fraction: {self.grounded_fraction:.3f}",
            f"- judge win-rate: {self.judge_winrate:.3f}",
            "",
            "| category | n | tool-call | grounded | judge |",
            "| --- | --- | --- | --- | --- |",
        ]
        for cat, m in sorted(self.per_category.items()):
            lines.append(
                f"| {cat} | {int(m['n'])} | {m['tool_call']:.3f} | "
                f"{m['grounded']:.3f} | {m['judge']:.3f} |"
            )
        return "\n".join(lines)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def score_run(
    cases, provider, judge, tool_specs, dispatch, bar: ShipBar = DEFAULT_BAR
) -> Scorecard:
    tc: list[float] = []
    gr: list[bool] = []
    verdicts: list[str] = []
    by_cat: dict[str, dict[str, list]] = defaultdict(lambda: {"tc": [], "gr": [], "v": []})

    for case in cases:
        run = run_candidate(case.user_prompt, provider, Budget(), tool_specs, dispatch)
        score = tool_call_score(run.tool_calls, case.gold_tool_calls)
        is_gr = grounded(run)
        verdict = judge.compare(case.user_prompt, case.gold_answer, run.final_answer)
        tc.append(score)
        gr.append(is_gr)
        verdicts.append(verdict)
        cat = by_cat[case.category]
        cat["tc"].append(score)
        cat["gr"].append(is_gr)
        cat["v"].append(verdict)

    per_category = {
        cat: {
            "n": len(v["tc"]),
            "tool_call": _mean(v["tc"]),
            "grounded": _mean([float(x) for x in v["gr"]]),
            "judge": win_rate(v["v"]),
        }
        for cat, v in by_cat.items()
    }
    return Scorecard(
        n=len(cases),
        tool_call_accuracy=_mean(tc),
        grounded_fraction=_mean([float(x) for x in gr]),
        judge_winrate=win_rate(verdicts),
        per_category=per_category,
    )
