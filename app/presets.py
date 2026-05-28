"""Named regime windows for one-click backtests in the GUI.

Each preset has a date range and a list of asset classes it applies to,
so the sidebar can hide buttons that don't make sense for the currently
selected model (e.g. don't show the COVID-crash button for a crypto
model whose data starts in 2021).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Preset:
    label: str
    start: date
    end: date
    applies_to: tuple[str, ...]  # asset classes — must overlap the model's assets


PRESETS: tuple[Preset, ...] = (
    Preset(
        label="Pre-COVID flat",
        start=date(2019, 6, 1),
        end=date(2020, 2, 15),
        applies_to=("equities",),
    ),
    Preset(
        label="COVID crash",
        start=date(2020, 2, 15),
        end=date(2020, 4, 30),
        applies_to=("equities",),
    ),
    Preset(
        label="2022 bear",
        start=date(2022, 1, 1),
        end=date(2022, 12, 31),
        applies_to=("equities",),
    ),
    Preset(
        label="2023-24 recovery",
        start=date(2023, 1, 1),
        end=date(2024, 12, 31),
        applies_to=("equities", "crypto"),
    ),
    Preset(
        label="BTC 2024-08-05 crash",
        start=date(2024, 8, 4),
        end=date(2024, 8, 6),
        applies_to=("crypto",),
    ),
)


def presets_for_assets(assets: list[str] | tuple[str, ...] | None) -> list[Preset]:
    """Return the presets that overlap the selected model's asset classes."""
    asset_set = set(assets or [])
    return [p for p in PRESETS if asset_set & set(p.applies_to)]
