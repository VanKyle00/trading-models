# webapp/events.py
"""Notable market-event date-range presets, grouped by asset class.

The config form offers these as quick "jump to this window" presets. Which set
appears depends on the selected model's asset class (``ModelSpec.assets``), so
equities models surface equity selloffs and crypto models surface crypto events.
Curate by editing the lists below — nothing else needs to change.
"""

from __future__ import annotations

from typing import TypedDict


class EventPreset(TypedDict):
    label: str
    start: str  # ISO date (YYYY-MM-DD)
    end: str


_EQUITIES: list[EventPreset] = [
    {"label": "COVID Crash", "start": "2020-02-19", "end": "2020-04-30"},
    {"label": "2022 Bear Market", "start": "2022-01-03", "end": "2022-10-14"},
    {"label": "SVB / Regional Bank Crisis", "start": "2023-03-01", "end": "2023-03-31"},
    {"label": "2018 Q4 Selloff", "start": "2018-10-01", "end": "2018-12-24"},
    {"label": "GFC 2008", "start": "2008-09-02", "end": "2009-03-31"},
    {"label": "Aug 2024 Vol Spike", "start": "2024-07-15", "end": "2024-08-30"},
]

_CRYPTO: list[EventPreset] = [
    {"label": "COVID Crash", "start": "2020-02-19", "end": "2020-04-30"},
    {"label": "LUNA / UST Collapse", "start": "2022-05-05", "end": "2022-05-20"},
    {"label": "FTX Collapse", "start": "2022-11-02", "end": "2022-11-21"},
    {"label": "2021 Peak → Crash", "start": "2021-11-08", "end": "2022-01-24"},
    {"label": "Spot ETF Rally", "start": "2023-10-01", "end": "2024-03-14"},
    {"label": "2018 Crypto Winter", "start": "2018-01-01", "end": "2018-12-15"},
]

_BY_ASSET: dict[str, list[EventPreset]] = {"equities": _EQUITIES, "crypto": _CRYPTO}


def events_for_assets(assets: tuple[str, ...]) -> list[EventPreset]:
    """Presets for the first recognised asset class; empty list if none match."""
    for asset in assets:
        if asset in _BY_ASSET:
            return _BY_ASSET[asset]
    return []
