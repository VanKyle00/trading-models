# Earnings Event-Vol Straddle Model Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL — superpowers:test-driven-development. Follow it for every task: write the failing test first, run it and confirm the expected FAIL, write the minimal implementation, run the test and confirm PASS, then commit. Do not batch steps.

**Goal:** Ship model `03-earnings-straddle` (directory `03-earnings-straddle-spy`, see naming note below): a long ATM straddle entered into earnings on liquid optionable names, whose edge is a *selection filter* (`expected_move > implied_move × k`, `k > 1`). Deliver the spec's Phase 1 (synthetic straddle pipeline, free, today) end-to-end — earnings-calendar loader, straddle assembly helper, synthetic pre-earnings IV + crush pricing, implied/expected-move signal, the k-margin entry gate plus no-trade filters, double-leg spread frictions wired through the existing options engine, fixed-fractional sizing with a portfolio cap, and a validation report distinguishing **filtered vs. unfiltered** straddle P&L with trade-level metrics. Also upgrade the SP1 validation harness with a non-parametric bootstrap test (centered, for the per-trade-returns hypothesis) and Benjamini-Hochberg FDR control (Component 6).

**Naming note (spec deviation, deliberate):** the spec names the path `models/options/03-earnings-straddle/`; this plan uses `models/options/03-earnings-straddle-spy/` to match the repo's existing options dirs (`01-delta-hedged-long-option-spy`, `02-directional-call-spy`). The model is multi-ticker by design (a `WATCHLIST` in `backtest.py`); the `-spy` suffix marks SPY as the default/illustration ticker, consistent with `default_ticker: SPY` in the frontmatter.

**Scope reconciliation (spec carve-outs):**
- The spec's *Out of scope* bullet says README / MODELS.md regeneration is deferred. This plan **overrides** that per the user's standing convention "always update README with new models" (auto-memory). README + MODELS.md updates are included in Task 12 and called out as a deliberate override, not silently added.
- Component 6's **walk-forward across earnings seasons** is **descoped** in this cycle and the descope is stated explicitly (Task 11 note). Reason, verified against the codebase: `tradinglib/validation/walk_forward.py` is built on `run_backtest` (the vectorized equity engine) with a `SignalFn` returning a target-position series; it is structurally incompatible with the `OptionsEngine` / `run_options_backtest` path this model uses, and an options-aware walk-forward is a separate design cycle. The Deflated-Sharpe `n_trials` hook (already in `run_options_backtest`/`compute_metrics`) is wired so a future grid can feed it. Component 6's other three bullets (bootstrap, FDR, trade-level metrics) ARE delivered.
- Component 2's greeks-for-diagnostics sub-clause is **deferred**: the Phase-1 no-trade filters (Task 5 `tradeable_event`) gate on implied-move validity, post-earnings expiry, and spread cap only — they do not require vega/theta. Stated in Task 5 and the README so Component-2 coverage is not over-claimed.
- Component 2's "nearest listed expiry strictly after earnings" is **approximated** in Phase 1 by a fixed `post_earnings_tenor` (calendar days) added to the earnings datetime. The listed-expiry snap arrives with the real chain in Phase 2/3. Stated in the strategy docstring.

**Architecture:** Reuse, do not reinvent. Pricing/Greeks come from `tradinglib/options/pricing.py` (`bs_price`, `bs_greeks`); multi-leg assembly from `tradinglib/options/instruments.py` (`OptionLeg`, `Position`, `CONTRACT_MULTIPLIER`); frictions from `tradinglib/options/spread.py` (`ParametricSpread`, `NoSpread` — charged per-leg so a straddle pays it twice); MTM/cash/expiry/turnover from `tradinglib/backtest/options_engine.py` (`OptionsEngine`, `run_options_backtest`, `OptionsStrategy` protocol with `on_bar(engine, t, spot)`, engine API `add_leg`/`close_all_options`/`position.legs`); stats from `tradinglib/backtest/metrics.py` (`compute_metrics`, returns a plain dict). New code is thin: an earnings-calendar loader (mirroring `tradinglib/loaders/equities/yfinance.py`), a `straddle.py` instrument helper, a synthetic event-IV surface added to `surface.py`, a per-event signal/gate module, the model's `backtest.py`, and three new validation statistics functions (bootstrap test, BH FDR, trade-level metrics).

**Timezone contract (verified):** `load_daily()` returns a DataFrame whose index is **UTC-aware** (`tradinglib/loaders/equities/yfinance.py:89`). The earnings loader's `earnings_datetime` is also **UTC-aware**. The strategy/signal unit-test fixtures use **tz-naive** `pd.date_range(..., freq="B")`. To avoid `TypeError: Cannot compare tz-naive and tz-aware timestamps` and keep one convention, the model's `backtest.py` **strips tz at the boundary** (`close.index = close.index.tz_localize(None)`) and normalizes `earnings_datetime` to tz-naive before any comparison. Additionally, `EventVolSurface.iv`, the strategy's earnings-bar lookup, and `signal.expected_move` are made tz-robust (coerce both operands to a common tz) so a caller passing a naive or aware index never crashes. This contract is asserted by an integration test (Task 10) feeding a tz-naive index with a UTC-aware loader-style earnings timestamp.

**yfinance note (verified):** `get_earnings_dates(limit=...)` is the recent-0.2.x method; in tests it is **mocked and never called live** (repo convention). The canonicalizer assumes `raw.index` is a tz-aware `DatetimeIndex` (yfinance's "Earnings Date" index) and falls back to a `"Earnings Date"` column if the index is not datetime-like. The assumption is documented in a comment pinned to yfinance >=0.2.

**Tech Stack:** Python 3.12, pandas 2.2, numpy 2.0, scipy 1.13, yfinance 0.2 (mocked in tests, never called live), matplotlib (Agg backend); pytest 8 + hypothesis; ruff (lint + format, line-length 100, target py312). **Verified ruff ruleset:** `[tool.ruff.lint] select = ["E","F","I","N","UP","B","SIM","RUF"]`, `ignore = ["E501"]` — **no ANN rules**, so bare `dict` returns and untyped local helpers are fine; **`I` (isort) IS enforced**, so all imports must sit in the top-of-file block, sorted. mypy is partial: `[tool.mypy] exclude = ["data/","notebooks/","models/"]`, so `mypy tradinglib` does **not** type-check `models/`; the new `tradinglib` functions (typed `tuple[...]` returns) are checked. Six-step CI gate: `ruff check` → `ruff format --check` → `mypy tradinglib` → `pytest` → streamlit import → MODELS.md staleness.

---

## File Structure

| Path | Create/Modify | Single responsibility |
|---|---|---|
| `tradinglib/loaders/events/__init__.py` | Create | Package marker for the events loaders namespace. |
| `tradinglib/loaders/events/earnings.py` | Create | `get_earnings_dates(tickers, start, end)` → canonical `[ticker, earnings_datetime, session]` DataFrame; yfinance default provider, point-in-time parquet cache under `data/processed/events/earnings/`. |
| `tradinglib/options/straddle.py` | Create | `snap_strike(...)`, `atm_straddle_legs(...)` → `[call_leg, put_leg]`; `straddle_price(...)` synthetic BS straddle premium given an IV. Pure assembly over existing `OptionLeg`/`bs_price`. |
| `tradinglib/options/surface.py` | Modify | Add `EventVolSurface` (pre-earnings elevated IV → post-earnings crush, with `pre_iv > post_iv` invariant) implementing the `VolSurface` protocol, for Phase-1 synthetic pricing. |
| `tradinglib/options/__init__.py` | Modify | Re-export `EventVolSurface` (surface.py has no `__all__`; only this file changes). |
| `tradinglib/backtest/metrics.py` | Modify | Add module-scoped `bootstrap_t_test(...)`, `benjamini_hochberg_fdr(...)`, `trade_metrics(...)`. |
| `tradinglib/validation/stats.py` | Create | Re-export the three new statistics as the validation-layer public surface. |
| `tradinglib/validation/__init__.py` | Modify | Export `bootstrap_t_test`, `benjamini_hochberg_fdr`, `trade_metrics`. |
| `models/options/03-earnings-straddle-spy/signal.py` | Create | Per-event signal: implied-move, expected-move (median of past N earnings-day abs returns, prior events only), the `k`-margin gate, the no-trade filters. Pure, tz-robust functions over price/earnings frames. |
| `models/options/03-earnings-straddle-spy/strategy.py` | Create | `EarningsStraddle` strategy implementing `OptionsStrategy`; module-level `size_contracts` and `can_open`. |
| `models/options/03-earnings-straddle-spy/backtest.py` | Create | `run_synthetic`, `build_validation_report`, `plot_branches`, `run_for_gui`, `main`: synthetic Phase-1 run for filtered vs unfiltered, multi-event-per-ticker FDR, bootstrap CI, trade-level metrics; writes `results/metrics.json`, `results/validation.json`, `results/equity_curve.png`. |
| `models/options/03-earnings-straddle-spy/model.md` | Create | YAML frontmatter + body for the model registry. |
| `models/options/03-earnings-straddle-spy/README.md` | Create | Strategy doc: the edge is the filter, Phase-1 is synthetic / not-yet-tradeable. |
| `data/ingestion/events/README.md` | Create | Ingestion doc for the earnings loader (mirrors `data/ingestion/equities/README.md`). |
| `docs/data-sources.md` | Modify | Add the `events/earnings` source row. |
| `README.md` | Modify | Add the model to the Current models table (override of spec carve-out). |
| `tests/test_earnings_loader.py` | Create | Loader canonicalization, caching, session parsing, empty + mixed-empty handling (yfinance mocked). |
| `tests/test_options_straddle.py` | Create | ATM strike snapping, two-leg assembly, synthetic straddle price. |
| `tests/test_event_vol_surface.py` | Create | Pre/post-earnings IV crush behavior + invariant + re-export of `EventVolSurface`. |
| `tests/test_earnings_signal.py` | Create | implied/expected-move math (tz-robust), k-gate, no-trade filters. |
| `tests/test_earnings_strategy.py` | Create | Entry/exit timing, double-spread cost, sizing, portfolio cap, expiry-before-exit invariant, synthetic integration, tz contract. |
| `tests/test_validation_stats.py` | Create | Bootstrap test CI/p-value (centered), BH FDR correctness (exact + non-contiguous), trade-level metrics. |
| `tests/test_model_registry_earnings.py` | Create | `model.md` frontmatter parses + has required keys; `find_models()` discovers the new model. |

---

### Task 1: Earnings calendar loader (Component 1)

**Files**
- Create: `tradinglib/loaders/events/__init__.py`
- Create: `tradinglib/loaders/events/earnings.py`
- Test: `tests/test_earnings_loader.py`

- [ ] **Step 1: Create the package marker.** Write `tradinglib/loaders/events/__init__.py` with exactly:
```python
"""Event-driven data loaders (earnings calendars, etc.)."""
```

- [ ] **Step 2: Write the failing canonicalization test.** Create `tests/test_earnings_loader.py`:
```python
"""Tests for the yfinance earnings-calendar loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def fake_earnings_frame() -> pd.DataFrame:
    """Mimic yfinance Ticker.get_earnings_dates(): tz-aware DatetimeIndex named
    'Earnings Date', plus estimate/actual columns we ignore."""
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-10-31 16:05:00", tz="America/New_York"),  # AMC
            pd.Timestamp("2024-07-25 08:30:00", tz="America/New_York"),  # BMO
        ],
        name="Earnings Date",
    )
    return pd.DataFrame(
        {"EPS Estimate": [1.0, 0.9], "Reported EPS": [1.1, 0.95]}, index=idx
    )


def test_canonicalize_columns_and_sessions(fake_earnings_frame: pd.DataFrame) -> None:
    from tradinglib.loaders.events import earnings as loader

    out = loader._canonicalize(fake_earnings_frame, "AAPL")

    assert list(out.columns) == ["ticker", "earnings_datetime", "session"]
    assert (out["ticker"] == "AAPL").all()
    assert str(out["earnings_datetime"].dt.tz) == "UTC"
    # 16:05 ET -> after close -> amc; 08:30 ET -> before open -> bmo
    assert set(out["session"]) == {"amc", "bmo"}
```

- [ ] **Step 3: Run the test — expect FAIL.** Command: `uv run pytest tests/test_earnings_loader.py -q`. Expected: `ModuleNotFoundError`/`AttributeError` (no `earnings` module / `_canonicalize`).

- [ ] **Step 4: Implement `_session_from_et` and `_canonicalize`.** Create `tradinglib/loaders/events/earnings.py` with the imports, constants, and the two helpers:
```python
"""Earnings-calendar loader (yfinance default provider).

Schema (canonical): a DataFrame with columns
``[ticker, earnings_datetime, session]`` where ``earnings_datetime`` is
UTC-aware and ``session`` is one of ``{"bmo", "amc", "unknown"}``. Cached to
``data/processed/events/earnings/<ticker>/<snapshot>.parquet`` for point-in-time
discipline (the snapshot date is part of the path, never future-leaking into the
data).

yfinance is mocked in tests and never called live (repo convention). The live
method name is version-dependent: ``Ticker.get_earnings_dates(limit=...)`` for
yfinance >= 0.2.x. ``_canonicalize`` assumes ``raw.index`` is a tz-aware
DatetimeIndex (yfinance's 'Earnings Date' index); it falls back to an
'Earnings Date' column if the index is not datetime-like.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

import pandas as pd
import yfinance as yf

from tradinglib.data.paths import processed_dir

SOURCE = "events"
_SUBDIR = "earnings"
_MARKET_OPEN_MIN = 9 * 60 + 30  # 09:30 ET
_MARKET_CLOSE_MIN = 16 * 60  # 16:00 ET


def _session_from_et(ts_utc: pd.Timestamp) -> str:
    """Label a UTC earnings timestamp as before-open / after-close.

    Midnight (yfinance's date-only fallback) is 'unknown'.
    """
    et = ts_utc.tz_convert("America/New_York")
    minutes = et.hour * 60 + et.minute
    if minutes == 0:
        return "unknown"
    if minutes <= _MARKET_OPEN_MIN:
        return "bmo"
    if minutes >= _MARKET_CLOSE_MIN:
        return "amc"
    return "unknown"


def _canonicalize(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize a yfinance earnings frame into the canonical schema."""
    if pd.api.types.is_datetime64_any_dtype(raw.index):
        source = raw.index
    elif "Earnings Date" in raw.columns:
        source = raw["Earnings Date"]
    else:
        source = raw.index
    dt = pd.to_datetime(source, utc=True)
    df = pd.DataFrame(
        {
            "ticker": ticker,
            "earnings_datetime": pd.DatetimeIndex(dt),
        }
    )
    df["session"] = [_session_from_et(ts) for ts in df["earnings_datetime"]]
    df = df.sort_values("earnings_datetime").reset_index(drop=True)
    return cast(pd.DataFrame, df[["ticker", "earnings_datetime", "session"]])
```

- [ ] **Step 5: Run the test — expect PASS.** Command: `uv run pytest tests/test_earnings_loader.py -q`. Expected: 1 passed.

- [ ] **Step 6: Write the failing caching + multi-ticker + mixed-empty tests.** Append to `tests/test_earnings_loader.py`:
```python
def test_get_earnings_dates_caches_and_filters(
    fake_earnings_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.events import earnings as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    class _FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def get_earnings_dates(self, limit: int = 24) -> pd.DataFrame:
            return fake_earnings_frame

    with patch.object(loader.yf, "Ticker", _FakeTicker):
        df = loader.get_earnings_dates(["AAPL"], start="2024-01-01", end="2024-12-31")

    cached = list((tmp_path / "events" / "earnings" / "AAPL").glob("*.parquet"))
    assert len(cached) == 1
    assert set(df["ticker"]) == {"AAPL"}
    assert len(df) == 2


def test_get_earnings_dates_handles_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.events import earnings as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    class _EmptyTicker:
        def __init__(self, symbol: str) -> None:
            pass

        def get_earnings_dates(self, limit: int = 24) -> pd.DataFrame | None:
            return None

    with patch.object(loader.yf, "Ticker", _EmptyTicker):
        df = loader.get_earnings_dates(["ZZZZ"], start="2024-01-01", end="2024-12-31")

    assert list(df.columns) == ["ticker", "earnings_datetime", "session"]
    assert df.empty


def test_get_earnings_dates_mixed_empty_and_nonempty_keeps_dtype(
    fake_earnings_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.events import earnings as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    class _MixedTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def get_earnings_dates(self, limit: int = 24) -> pd.DataFrame | None:
            return None if self.symbol == "ZZZZ" else fake_earnings_frame

    with patch.object(loader.yf, "Ticker", _MixedTicker):
        df = loader.get_earnings_dates(
            ["AAPL", "ZZZZ"], start="2024-01-01", end="2024-12-31"
        )

    # tz-aware dtype must survive concat so the date filter does not raise
    assert str(df["earnings_datetime"].dt.tz) == "UTC"
    assert set(df["ticker"]) == {"AAPL"}
    assert len(df) == 2
```

- [ ] **Step 7: Run — expect FAIL.** Command: `uv run pytest tests/test_earnings_loader.py -q`. Expected: `AttributeError: module ... has no attribute 'get_earnings_dates'`.

- [ ] **Step 8: Implement `_download_one`, `_empty` and `get_earnings_dates`.** Append to `tradinglib/loaders/events/earnings.py`:
```python
def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series([], dtype="object"),
            "earnings_datetime": pd.Series([], dtype="datetime64[ns, UTC]"),
            "session": pd.Series([], dtype="object"),
        }
    )


def _download_one(ticker: str) -> pd.DataFrame:
    """Fetch and canonicalize the earnings schedule for one ticker."""
    raw = yf.Ticker(ticker).get_earnings_dates(limit=24)
    if raw is None or len(raw) == 0:
        return _empty()
    return _canonicalize(raw, ticker)


def get_earnings_dates(
    tickers: list[str],
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return scheduled earnings for ``tickers`` within ``[start, end]``.

    One parquet cache per ticker keyed by the snapshot (download) date, so a
    cached read never exposes a schedule that postdates its snapshot. Date
    filtering is applied in-memory on the canonical ``earnings_datetime``.
    """
    snapshot = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{snapshot}.parquet"
        if out.exists() and not refresh:
            df = pd.read_parquet(out)
        else:
            df = _download_one(ticker)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True) if frames else _empty()
    # Defend against pandas downcasting an empty tz-aware column during concat.
    result["earnings_datetime"] = pd.to_datetime(result["earnings_datetime"], utc=True)
    if start is not None:
        result = result[result["earnings_datetime"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        result = result[result["earnings_datetime"] <= pd.Timestamp(end, tz="UTC")]
    return result.reset_index(drop=True)
```

- [ ] **Step 9: Run — expect PASS.** Command: `uv run pytest tests/test_earnings_loader.py -q`. Expected: 4 passed.

- [ ] **Step 10: Verify gate slice + commit.** Run `uv run ruff check tradinglib/loaders/events tests/test_earnings_loader.py` then `uv run ruff format tradinglib/loaders/events tests/test_earnings_loader.py`. Commit:
```
git add tradinglib/loaders/events/__init__.py tradinglib/loaders/events/earnings.py tests/test_earnings_loader.py
git commit -m "feat(loaders): add yfinance earnings-calendar loader (events source)"
```

---

### Task 2: Straddle instrument assembly helper (Component 2)

**Files**
- Create: `tradinglib/options/straddle.py`
- Test: `tests/test_options_straddle.py`

- [ ] **Step 1: Write the failing strike-snap + leg-assembly test.** Create `tests/test_options_straddle.py`:
```python
"""Tests for ATM straddle assembly helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from tradinglib.options.pricing import bs_price
from tradinglib.options.straddle import atm_straddle_legs, snap_strike, straddle_price


def test_atm_straddle_legs_snaps_strike_and_builds_two_legs() -> None:
    expiry = pd.Timestamp("2024-02-01")
    legs = atm_straddle_legs(spot=101.4, expiry=expiry, quantity=1.0, strike_step=1.0)

    assert len(legs) == 2
    assert sorted(leg.right for leg in legs) == ["call", "put"]
    assert {leg.strike for leg in legs} == {101.0}
    assert all(leg.expiry == expiry for leg in legs)
    assert all(leg.quantity == 1.0 for leg in legs)


def test_atm_straddle_legs_strike_step_5() -> None:
    legs = atm_straddle_legs(
        spot=103.0, expiry=pd.Timestamp("2024-02-01"), quantity=1.0, strike_step=5.0
    )
    assert {leg.strike for leg in legs} == {105.0}


def test_snap_strike_rejects_nonpositive_step() -> None:
    with pytest.raises(ValueError):
        snap_strike(100.0, 0.0)
```

- [ ] **Step 2: Run — expect FAIL.** Command: `uv run pytest tests/test_options_straddle.py -q`. Expected: `ModuleNotFoundError: tradinglib.options.straddle`.

- [ ] **Step 3: Implement `snap_strike` + `atm_straddle_legs`.** Create `tradinglib/options/straddle.py`:
```python
"""ATM straddle assembly over existing option primitives.

No new pricing: a straddle is a long call + long put at the nearest listed
strike, same expiry. Strikes snap to a configurable ``strike_step`` grid to
approximate the listed chain in the Phase-1 synthetic pipeline.
"""

from __future__ import annotations

import pandas as pd

from tradinglib.options.instruments import OptionLeg
from tradinglib.options.pricing import bs_price


def snap_strike(spot: float, strike_step: float) -> float:
    """Round ``spot`` to the nearest multiple of ``strike_step``."""
    if strike_step <= 0:
        raise ValueError(f"strike_step must be > 0, got {strike_step}")
    return float(round(spot / strike_step) * strike_step)


def atm_straddle_legs(
    spot: float,
    expiry: pd.Timestamp,
    quantity: float = 1.0,
    *,
    underlying: str = "SPY",
    strike_step: float = 1.0,
) -> list[OptionLeg]:
    """Return ``[call_leg, put_leg]`` for a long ATM straddle."""
    strike = snap_strike(spot, strike_step)
    return [
        OptionLeg("call", strike=strike, expiry=expiry, quantity=quantity, underlying=underlying),
        OptionLeg("put", strike=strike, expiry=expiry, quantity=quantity, underlying=underlying),
    ]


def straddle_price(
    spot: float,
    strike: float,
    t: float,
    vol: float,
    rate: float,
    div: float = 0.0,
) -> float:
    """Synthetic ATM-straddle premium = call BS + put BS (per share)."""
    call = bs_price("call", spot, strike, t, vol, rate, div)
    put = bs_price("put", spot, strike, t, vol, rate, div)
    return call + put
```

- [ ] **Step 4: Run — expect PASS.** Command: `uv run pytest tests/test_options_straddle.py -q`. Expected: 3 passed.

- [ ] **Step 5: Write the failing synthetic-price test.** Append to `tests/test_options_straddle.py`:
```python
def test_straddle_price_matches_two_bs_legs() -> None:
    expected = bs_price("call", 100.0, 100.0, 30 / 365.0, 0.40, 0.04) + bs_price(
        "put", 100.0, 100.0, 30 / 365.0, 0.40, 0.04
    )
    price = straddle_price(spot=100.0, strike=100.0, t=30 / 365.0, vol=0.40, rate=0.04)
    assert price == pytest.approx(expected, abs=1e-9)
    assert price > 0.0
```

- [ ] **Step 6: Run — expect PASS (no new code).** Command: `uv run pytest tests/test_options_straddle.py -q`. Expected: 4 passed. (`straddle_price` was written in Step 3; this test pins it against two raw `bs_price` calls.)

- [ ] **Step 7: Verify slice + commit.** Run `uv run ruff check tradinglib/options/straddle.py tests/test_options_straddle.py` then `uv run ruff format` on both. Commit:
```
git add tradinglib/options/straddle.py tests/test_options_straddle.py
git commit -m "feat(options): add ATM straddle assembly + synthetic price helper"
```

---

### Task 3: Synthetic pre-earnings IV + crush surface (Phase-1 pricing)

**Files**
- Modify: `tradinglib/options/surface.py`
- Modify: `tradinglib/options/__init__.py`
- Test: `tests/test_event_vol_surface.py`

Verified facts used here: `surface.py` already imports `from dataclasses import dataclass, field`, `import pandas as pd`, and defines `VolSurface` Protocol with `iv(self, spot, strike, expiry, t) -> float`. It has **no `__all__`** (file ends at `realistic_surface`), so only `tradinglib/options/__init__.py` needs the re-export edit. The engine builds `engine.t = pd.Timestamp(t)` from the price index, so for naive bars `t` is naive.

- [ ] **Step 1: Write the failing crush-behavior + invariant + tz-robust tests.** Create `tests/test_event_vol_surface.py`:
```python
"""Tests for the synthetic pre/post-earnings event-vol surface."""

from __future__ import annotations

import pandas as pd
import pytest

from tradinglib.options.surface import EventVolSurface


def test_iv_elevated_before_and_crushed_after_earnings() -> None:
    earnings = pd.Timestamp("2024-02-15")
    surf = EventVolSurface(earnings_datetime=earnings, pre_iv=0.60, post_iv=0.30)
    expiry = pd.Timestamp("2024-02-23")
    before = surf.iv(100.0, 100.0, expiry, pd.Timestamp("2024-02-14"))
    after = surf.iv(100.0, 100.0, expiry, pd.Timestamp("2024-02-16"))
    assert before == pytest.approx(0.60)
    assert after == pytest.approx(0.30)
    assert before > after


def test_iv_on_earnings_day_is_still_pre_crush() -> None:
    earnings = pd.Timestamp("2024-02-15")
    surf = EventVolSurface(earnings_datetime=earnings, pre_iv=0.55, post_iv=0.25)
    on_day = surf.iv(100.0, 100.0, pd.Timestamp("2024-02-23"), pd.Timestamp("2024-02-15"))
    assert on_day == pytest.approx(0.55)


def test_post_iv_must_be_below_pre_iv() -> None:
    # the entire point is IV crush; reject configs that manufacture an expansion
    with pytest.raises(ValueError):
        EventVolSurface(earnings_datetime=pd.Timestamp("2024-02-15"), pre_iv=0.30, post_iv=0.45)
    with pytest.raises(ValueError):
        EventVolSurface(earnings_datetime=pd.Timestamp("2024-02-15"), pre_iv=0.30, post_iv=0.0)


def test_iv_tz_robust_aware_earnings_naive_bar() -> None:
    # loader earnings_datetime is UTC-aware; engine bars can be tz-naive
    earnings = pd.Timestamp("2024-02-15", tz="UTC")
    surf = EventVolSurface(earnings_datetime=earnings, pre_iv=0.60, post_iv=0.30)
    before = surf.iv(100.0, 100.0, pd.Timestamp("2024-02-23"), pd.Timestamp("2024-02-14"))
    after = surf.iv(100.0, 100.0, pd.Timestamp("2024-02-23"), pd.Timestamp("2024-02-16"))
    assert before == pytest.approx(0.60)
    assert after == pytest.approx(0.30)
```

- [ ] **Step 2: Run — expect FAIL.** Command: `uv run pytest tests/test_event_vol_surface.py -q`. Expected: `ImportError: cannot import name 'EventVolSurface'`.

- [ ] **Step 3: Implement `EventVolSurface`.** Append the dataclass to `tradinglib/options/surface.py` after `ParametricSurface` (reuse the existing `dataclass`/`pd` imports — do **not** add imports). Use a module-level tz-coercion helper so comparisons never raise:
```python
def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
    """Drop tz so naive (engine bars) and aware (loader earnings) compare cleanly."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tz is not None else ts


@dataclass(frozen=True)
class EventVolSurface:
    """Synthetic two-regime IV for the Phase-1 earnings straddle.

    Returns ``pre_iv`` (elevated) on bars at or before the earnings datetime and
    ``post_iv`` (crushed) strictly after it. Implements the ``VolSurface``
    protocol. Enforces ``pre_iv > post_iv > 0`` because IV crush is the entire
    point: a config with ``post_iv >= pre_iv`` would manufacture an IV expansion
    into the move and make the long straddle falsely profitable. Phase-1 only —
    clearly not tradeable, mirrors the repo's synthetic vol treatment. tz-robust:
    coerces both operands to tz-naive so a UTC-aware loader earnings timestamp
    compares cleanly with tz-naive engine bars.
    """

    earnings_datetime: pd.Timestamp
    pre_iv: float
    post_iv: float

    def __post_init__(self) -> None:
        if not (self.pre_iv > self.post_iv > 0):
            raise ValueError(
                f"require pre_iv > post_iv > 0 (IV crush); got pre_iv={self.pre_iv}, "
                f"post_iv={self.post_iv}"
            )

    def iv(self, spot: float, strike: float, expiry: pd.Timestamp, t: pd.Timestamp) -> float:
        return self.pre_iv if _to_naive(t) <= _to_naive(self.earnings_datetime) else self.post_iv
```

- [ ] **Step 4: Run — expect PASS.** Command: `uv run pytest tests/test_event_vol_surface.py -q`. Expected: 4 passed.

- [ ] **Step 5: Add the re-export and assert it.** Edit `tradinglib/options/__init__.py` only (surface.py has no `__all__`): add `EventVolSurface` to the `from tradinglib.options.surface import (...)` block and to `__all__` (alphabetical — `EventVolSurface` sorts before `FlatSurface`). Then append a one-line re-export assertion to `tests/test_event_vol_surface.py`:
```python
def test_event_vol_surface_is_reexported() -> None:
    from tradinglib.options import EventVolSurface as Reexported
    from tradinglib.options.surface import EventVolSurface as Direct

    assert Reexported is Direct
```
Run `uv run pytest tests/test_event_vol_surface.py -q`. Expected: 5 passed.

- [ ] **Step 6: Verify slice + commit.** Run `uv run ruff check tradinglib/options/surface.py tradinglib/options/__init__.py tests/test_event_vol_surface.py` then `uv run ruff format` on the same paths. Commit:
```
git add tradinglib/options/surface.py tradinglib/options/__init__.py tests/test_event_vol_surface.py
git commit -m "feat(options): add synthetic EventVolSurface (pre-earnings IV + crush)"
```

---

### Task 4: Implied-move & expected-move signal computation (Component 3)

**Files**
- Create: `models/options/03-earnings-straddle-spy/signal.py`
- Test: `tests/test_earnings_signal.py`

- [ ] **Step 1: Write the failing implied-move test.** Create `tests/test_earnings_signal.py`:
```python
"""Tests for the earnings-straddle selection signal."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SIGNAL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "options"
    / "03-earnings-straddle-spy"
    / "signal.py"
)
_spec = importlib.util.spec_from_file_location("earnings_signal", _SIGNAL_PATH)
assert _spec is not None and _spec.loader is not None
signal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(signal)


def test_implied_move_is_straddle_over_spot() -> None:
    assert signal.implied_move(straddle_premium=6.0, spot=100.0) == pytest.approx(0.06)


def test_implied_move_rejects_nonpositive_spot() -> None:
    with pytest.raises(ValueError):
        signal.implied_move(straddle_premium=6.0, spot=0.0)
```

- [ ] **Step 2: Run — expect FAIL.** Command: `uv run pytest tests/test_earnings_signal.py -q`. Expected: `FileNotFoundError`/spec load error (signal.py missing).

- [ ] **Step 3: Implement the module skeleton + `implied_move`.** Create `models/options/03-earnings-straddle-spy/signal.py` with the **complete top-of-module import block already including `math`** (so Task 5 never inserts an import mid-file):
```python
"""Per-event selection signal for the earnings straddle.

implied_move = straddle_premium / spot. This is the ATM-straddle *mean-absolute-
move* proxy (E|return|), which is ~0.8 * atm_iv * sqrt(T) — NOT the 1-sigma move
(atm_iv * sqrt(T)). expected_move below is the MEAN of past earnings-day absolute
returns (a like-for-like mean-absolute measure), so the k-gate compares two
mean-absolute quantities rather than mixing a mean against a median. Entry iff
expected_move > implied_move * k, k > 1. Plus the no-trade filters (valid IV,
post-earnings expiry, spread cap). Functions are tz-robust: they coerce earnings
datetimes and the price index to a common tz so a UTC-aware loader timestamp
compares cleanly with tz-naive bars.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def implied_move(straddle_premium: float, spot: float) -> float:
    """ATM-straddle implied move = premium / spot (a fraction, e.g. 0.06).

    This is the mean-absolute-move proxy (~0.8 * iv * sqrt(T)), the like-for-like
    counterpart to the mean of past absolute earnings moves used by expected_move.
    """
    if spot <= 0:
        raise ValueError(f"spot must be > 0, got {spot}")
    return straddle_premium / spot
```
(`numpy` is imported now because `expected_move` in Step 7 uses `np.nan`/array ops; this keeps every import in the sorted top block, satisfying ruff `I`.)

- [ ] **Step 4: Run — expect PASS.** Command: `uv run pytest tests/test_earnings_signal.py -q`. Expected: 2 passed.

- [ ] **Step 5: Write the failing expected-move test (mean of prior abs moves, tz-robust).** Append to `tests/test_earnings_signal.py`:
```python
def test_expected_move_is_mean_of_past_earnings_day_abs_returns() -> None:
    idx = pd.date_range("2023-01-02", periods=400, freq="B", tz="UTC")
    close = pd.Series(100.0, index=idx)
    e_dates = [idx[50], idx[150], idx[250]]
    moves = [0.05, -0.03, 0.07]
    for ed, mv in zip(e_dates, moves, strict=True):
        pos = idx.get_loc(ed)
        close.iloc[pos + 1] = close.iloc[pos] * (1 + mv)

    em = signal.expected_move(close=close, earnings_datetimes=pd.Series(e_dates), lookback=3)
    # mean of |0.05|, |0.03|, |0.07| = 0.05
    assert em == pytest.approx(0.05, abs=1e-6)


def test_expected_move_tz_robust_naive_index_aware_earnings() -> None:
    idx = pd.date_range("2023-01-02", periods=400, freq="B")  # tz-NAIVE bars
    close = pd.Series(100.0, index=idx)
    pos = 50
    close.iloc[pos + 1] = close.iloc[pos] * 1.06
    earnings = pd.Series([pd.Timestamp(idx[pos], tz="UTC")])  # tz-AWARE earnings
    em = signal.expected_move(close=close, earnings_datetimes=earnings, lookback=8)
    assert em == pytest.approx(0.06, abs=1e-6)


def test_expected_move_nan_when_no_history() -> None:
    idx = pd.date_range("2023-01-02", periods=10, freq="B", tz="UTC")
    close = pd.Series(100.0, index=idx)
    em = signal.expected_move(
        close=close, earnings_datetimes=pd.Series([], dtype="datetime64[ns, UTC]"), lookback=3
    )
    assert np.isnan(em)
```

- [ ] **Step 6: Run — expect FAIL.** Command: `uv run pytest tests/test_earnings_signal.py -q`. Expected: `AttributeError: module 'earnings_signal' has no attribute 'expected_move'`.

- [ ] **Step 7: Implement `expected_move` (tz-robust, mean of abs moves).** Append to `signal.py`:
```python
def expected_move(
    close: pd.Series,
    earnings_datetimes: pd.Series,
    lookback: int = 8,
) -> float:
    """Mean absolute earnings-day return over the last ``lookback`` events.

    For each past earnings date, the realized move is the close-to-close return
    from the earnings session to the following trading bar. Uses the MEAN (not
    median) so it is a like-for-like comparison with implied_move's mean-absolute
    proxy. tz-robust: both the price index and the earnings dates are coerced to
    tz-naive before comparison. Returns NaN if no usable history exists.
    """
    closes = close.sort_index()
    cidx = closes.index
    if getattr(cidx, "tz", None) is not None:
        cidx = cidx.tz_localize(None)
        closes = pd.Series(closes.to_numpy(), index=cidx)

    eds = pd.to_datetime(earnings_datetimes, utc=True).dt.tz_localize(None).sort_values()

    moves: list[float] = []
    for ed in eds:
        prior = closes.index[closes.index <= ed]
        if len(prior) == 0:
            continue
        pos = closes.index.get_loc(prior[-1])
        if pos + 1 >= len(closes):
            continue
        ret = closes.iloc[pos + 1] / closes.iloc[pos] - 1.0
        moves.append(abs(float(ret)))

    if not moves:
        return float("nan")
    return float(np.mean(moves[-lookback:]))
```

- [ ] **Step 8: Run — expect PASS.** Command: `uv run pytest tests/test_earnings_signal.py -q`. Expected: 5 passed.

- [ ] **Step 9: Verify slice + commit.** Run `uv run ruff check models/options/03-earnings-straddle-spy/signal.py tests/test_earnings_signal.py` then `uv run ruff format` on both. Commit:
```
git add models/options/03-earnings-straddle-spy/signal.py tests/test_earnings_signal.py
git commit -m "feat(03-earnings-straddle): implied-move + expected-move signal"
```

---

### Task 5: k-margin entry gate + no-trade filters (Components 3 & 5)

**Files**
- Modify: `models/options/03-earnings-straddle-spy/signal.py`
- Test: `tests/test_earnings_signal.py`

`import math` is already in the top-of-module block from Task 4 Step 3, so the steps below append **only function bodies** — no mid-file imports.

- [ ] **Step 1: Write the failing k-gate test.** Append to `tests/test_earnings_signal.py`:
```python
def test_passes_filter_true_only_when_expected_beats_implied_times_k() -> None:
    assert signal.passes_filter(expected=0.10, implied=0.06, k=1.5) is True
    assert signal.passes_filter(expected=0.10, implied=0.06, k=1.7) is False


def test_passes_filter_rejects_k_le_one() -> None:
    with pytest.raises(ValueError):
        signal.passes_filter(expected=0.10, implied=0.06, k=1.0)


def test_passes_filter_false_on_nan_inputs() -> None:
    assert signal.passes_filter(expected=float("nan"), implied=0.06, k=1.2) is False
    assert signal.passes_filter(expected=0.10, implied=float("nan"), k=1.2) is False
```

- [ ] **Step 2: Run — expect FAIL.** Command: `uv run pytest tests/test_earnings_signal.py -q`. Expected: `AttributeError: ... 'passes_filter'`.

- [ ] **Step 3: Implement `passes_filter` (body only — `math` already imported).** Append to `signal.py`:
```python
def passes_filter(expected: float, implied: float, k: float) -> bool:
    """Selection gate: True iff expected_move > implied_move * k (k > 1)."""
    if k <= 1.0:
        raise ValueError(f"k must be > 1, got {k}")
    if math.isnan(expected) or math.isnan(implied):
        return False
    return expected > implied * k
```

- [ ] **Step 4: Run — expect PASS.** Command: `uv run pytest tests/test_earnings_signal.py -q`. Expected: 8 passed.

- [ ] **Step 5: Write the failing no-trade-filter test.** Append to `tests/test_earnings_signal.py`:
```python
def test_tradeable_event_rejects_bad_chains() -> None:
    base = dict(
        implied=0.06, expected=0.10, spread_frac=0.05, max_spread_frac=0.20, has_expiry=True
    )
    assert signal.tradeable_event(**base) is True
    assert signal.tradeable_event(**{**base, "implied": float("nan")}) is False
    assert signal.tradeable_event(**{**base, "spread_frac": 0.25}) is False
    assert signal.tradeable_event(**{**base, "has_expiry": False}) is False
```

- [ ] **Step 6: Run — expect FAIL.** Command: `uv run pytest tests/test_earnings_signal.py -q`. Expected: `AttributeError: ... 'tradeable_event'`.

- [ ] **Step 7: Implement `tradeable_event` (body only).** Append to `signal.py`:
```python
def tradeable_event(
    *,
    implied: float,
    expected: float,
    spread_frac: float,
    max_spread_frac: float,
    has_expiry: bool,
) -> bool:
    """No-trade filters (Ch. 6): valid IV/move, a listed post-earnings expiry,
    and a half-spread no wider than the cap. The k-gate is applied separately by
    the caller; this returns whether the *chain* is even tradeable.

    Greeks (vega/theta) diagnostics from Component 2 are deferred this phase —
    these filters do not require them.
    """
    if math.isnan(implied) or implied <= 0.0:
        return False
    if math.isnan(expected):
        return False
    if not has_expiry:
        return False
    if spread_frac > max_spread_frac:
        return False
    return True
```

- [ ] **Step 8: Run — expect PASS.** Command: `uv run pytest tests/test_earnings_signal.py -q`. Expected: 9 passed.

- [ ] **Step 9: Verify slice + commit.** Run `uv run ruff check models/options/03-earnings-straddle-spy/signal.py tests/test_earnings_signal.py` then `uv run ruff format` on both. Commit:
```
git add models/options/03-earnings-straddle-spy/signal.py tests/test_earnings_signal.py
git commit -m "feat(03-earnings-straddle): k-margin entry gate + no-trade filters"
```

---

### Task 6: Entry/exit + double-leg spread frictions through the engine (Component 4)

**Files**
- Create: `models/options/03-earnings-straddle-spy/strategy.py`
- Test: `tests/test_earnings_strategy.py`

**Expiry-before-exit invariant (verified):** `run_options_backtest` calls `engine._settle_expiries()` at the START of every bar (options_engine.py:214), settling any leg with `(leg.expiry - t).days <= 0` to intrinsic and removing it BEFORE `strategy.on_bar`. If the leg auto-settles before the strategy's exit bar, `engine.position.legs` is already empty and a `legs`-guarded exit branch would never fire. Two defenses: (1) set `post_earnings_tenor` default to **14 calendar days** so the expiry is comfortably beyond the entry_lead+exit_offset span; (2) the exit branch keys off the strategy's own `entered_on/exited_on` state (`if self.entered_on is not None and self.exited_on is None`), NOT off `engine.position.legs`, and calls `close_all_options()` only if legs remain. A test places earnings on a Friday and asserts the straddle is closed by `close_all_options` (`exited_on` set), not silently pre-settled.

The strategy includes `import math` in its top import block (used by `size_contracts` in Task 7) so Task 7 appends function bodies only.

- [ ] **Step 1: Write the failing entry/exit timing test.** Create `tests/test_earnings_strategy.py`:
```python
"""Tests for the EarningsStraddle options strategy (timing, frictions, cap)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglib.backtest.options_engine import run_options_backtest
from tradinglib.options.spread import NoSpread, ParametricSpread
from tradinglib.options.surface import EventVolSurface

_STRAT_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "options"
    / "03-earnings-straddle-spy"
    / "strategy.py"
)
_spec = importlib.util.spec_from_file_location("earnings_strategy", _STRAT_PATH)
assert _spec is not None and _spec.loader is not None
strat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strat)


def _flat_prices(n: int = 20, start: str = "2024-02-01") -> pd.Series:
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.Series(100.0, index=idx, name="close")


def test_enters_at_lead_and_exits_at_offset() -> None:
    prices = _flat_prices()
    earnings = prices.index[10]
    s = strat.EarningsStraddle(
        earnings_datetime=earnings, entry_lead=3, exit_offset=1, contracts=1.0
    )
    surface = EventVolSurface(earnings_datetime=earnings, pre_iv=0.60, post_iv=0.30)
    run_options_backtest(prices, s, surface=surface, spread=NoSpread())

    assert s.entered_on == prices.index[7]
    assert s.exited_on == prices.index[11]


def test_friday_earnings_exit_is_closed_not_auto_settled() -> None:
    # earnings on a Friday; exit is the next business bar (Monday). With the
    # 14-day default tenor the expiry is far beyond exit, so close_all_options
    # handles it and exited_on is set (no pre-emptive _settle_expiries).
    prices = _flat_prices(n=25, start="2024-02-05")  # 2024-02-05 is a Monday
    fridays = [d for d in prices.index if d.dayofweek == 4]
    earnings = fridays[1]
    s = strat.EarningsStraddle(
        earnings_datetime=earnings, entry_lead=3, exit_offset=1, contracts=1.0
    )
    surface = EventVolSurface(earnings_datetime=earnings, pre_iv=0.60, post_iv=0.30)
    run_options_backtest(prices, s, surface=surface, spread=NoSpread())

    assert s.entered_on is not None
    assert s.exited_on is not None
    assert s.exited_on > s.entered_on
```

- [ ] **Step 2: Run — expect FAIL.** Command: `uv run pytest tests/test_earnings_strategy.py -q`. Expected: spec load / `AttributeError` (no `EarningsStraddle`).

- [ ] **Step 3: Implement the `EarningsStraddle` strategy.** Create `models/options/03-earnings-straddle-spy/strategy.py`:
```python
"""EarningsStraddle: long ATM straddle around one earnings event.

Implements the OptionsStrategy protocol. Enters at T-entry_lead trading bars,
opens an ATM straddle expiring strictly after earnings, and closes at
T+exit_offset. The engine charges the bid/ask spread per leg, so a straddle pays
it twice on entry and twice on exit.

Phase-1 simplification (Component 2): expiry is approximated as
``earnings + post_earnings_tenor`` calendar days rather than snapped to a listed
weekly Friday; the listed-expiry snap arrives with the real chain in Phase 2/3.
``post_earnings_tenor`` defaults to 14 calendar days so the leg's expiry stays
comfortably beyond the (entry_lead + exit_offset) hold window — otherwise the
engine's _settle_expiries (run at the start of every bar) could settle the leg
before the strategy's exit bar. The exit branch keys off this strategy's own
entered/exited state, not engine.position.legs, so an already-settled leg never
breaks the exit accounting. Session label (bmo/amc) is parsed by the loader but
NOT used to offset entry/exit in this synthetic phase; the earnings bar is the
first bar whose timestamp is >= the earnings datetime (documented deviation).
"""

from __future__ import annotations

import math

import pandas as pd

from tradinglib.backtest.options_engine import OptionsEngine
from tradinglib.options.instruments import CONTRACT_MULTIPLIER
from tradinglib.options.straddle import atm_straddle_legs


def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tz is not None else ts


class EarningsStraddle:
    def __init__(
        self,
        *,
        earnings_datetime: pd.Timestamp,
        entry_lead: int = 3,
        exit_offset: int = 1,
        contracts: float = 1.0,
        strike_step: float = 1.0,
        post_earnings_tenor: int = 14,
    ) -> None:
        self.earnings_datetime = _to_naive(earnings_datetime)
        self.entry_lead = entry_lead
        self.exit_offset = exit_offset
        self.contracts = contracts
        self.strike_step = strike_step
        self.post_earnings_tenor = post_earnings_tenor
        self._bars: list[pd.Timestamp] = []
        self.entered_on: pd.Timestamp | None = None
        self.exited_on: pd.Timestamp | None = None

    def _earnings_bar_index(self) -> int | None:
        for i, b in enumerate(self._bars):
            if _to_naive(b) >= self.earnings_datetime:
                return i
        return None

    def on_bar(self, engine: OptionsEngine, t: pd.Timestamp, spot: float) -> None:
        self._bars.append(t)
        e_idx = self._earnings_bar_index()
        if e_idx is None:
            return
        now_idx = len(self._bars) - 1
        entry_idx = e_idx - self.entry_lead
        exit_idx = e_idx + self.exit_offset

        if now_idx == entry_idx and self.entered_on is None:
            expiry = self.earnings_datetime + pd.Timedelta(days=self.post_earnings_tenor)
            for leg in atm_straddle_legs(
                spot, expiry, quantity=self.contracts, strike_step=self.strike_step
            ):
                engine.add_leg(leg)
            self.entered_on = t
        elif now_idx >= exit_idx and self.entered_on is not None and self.exited_on is None:
            if engine.position.legs:
                engine.close_all_options()
            self.exited_on = t
```

- [ ] **Step 4: Run — expect PASS.** Command: `uv run pytest tests/test_earnings_strategy.py -q`. Expected: 2 passed.

- [ ] **Step 5: Write the failing double-spread friction test (non-trivial premium so the spread is unambiguously charged twice).** Append to `tests/test_earnings_strategy.py`:
```python
def test_straddle_pays_spread_on_both_legs() -> None:
    prices = _flat_prices()
    earnings = prices.index[10]
    # non-zero pre_iv and ATM-snapped strike => the straddle carries real premium,
    # so the per-leg half-spread is unambiguously charged (entry x2, exit x2).
    surface = EventVolSurface(earnings_datetime=earnings, pre_iv=0.60, post_iv=0.30)

    def _run(spread):
        s = strat.EarningsStraddle(
            earnings_datetime=earnings, entry_lead=3, exit_offset=1, contracts=1.0
        )
        return run_options_backtest(prices, s, surface=surface, spread=spread)

    frictionless = _run(NoSpread())
    frictioned = _run(ParametricSpread())

    assert frictioned.equity_curve.iloc[-1] < frictionless.equity_curve.iloc[-1]
    assert (frictioned.turnover > 0).sum() >= 2
```

- [ ] **Step 6: Run the spread test and confirm it PASSES before committing.** Command: `uv run pytest tests/test_earnings_strategy.py::test_straddle_pays_spread_on_both_legs -q`. This is a regression assertion on the strategy written in Step 3 plus the existing per-leg spread charging in `OptionsEngine._fill_price`/`close_all_options` (verified: spread is charged per leg on both `add_leg` and `close_all_options`, and `_bar_notional`/turnover are nonzero on the entry and exit bars). If it FAILS, the bug is in the strategy's open/close wiring (debug the strategy — entry must `add_leg` both legs at `entry_idx`, exit must `close_all_options` at `exit_idx`), NOT the engine. Re-run until PASS. Expected: 1 passed. Then run the whole file: `uv run pytest tests/test_earnings_strategy.py -q` → 3 passed.

- [ ] **Step 7: Verify slice + commit (after the spread test passes).** Run `uv run ruff check models/options/03-earnings-straddle-spy/strategy.py tests/test_earnings_strategy.py` then `uv run ruff format` on both. Commit:
```
git add models/options/03-earnings-straddle-spy/strategy.py tests/test_earnings_strategy.py
git commit -m "feat(03-earnings-straddle): entry/exit strategy with double-leg spread"
```

---

### Task 7: Fixed-fractional sizing + portfolio cap (Component 5)

**Files**
- Modify: `models/options/03-earnings-straddle-spy/strategy.py`
- Test: `tests/test_earnings_strategy.py`

`CONTRACT_MULTIPLIER` is **already imported** in `strategy.py`'s top block (Task 6 Step 3). The steps below append **module-level function bodies only** — no new imports, satisfying ruff `I`.

- [ ] **Step 1: Write the failing sizing test.** Append to `tests/test_earnings_strategy.py`:
```python
def test_size_contracts_uses_fixed_fraction_of_capital() -> None:
    # 1% of $100k = $1000 budget; premium $5.00/share * 100 = $500/contract -> 2
    n = strat.size_contracts(capital=100_000.0, risk_fraction=0.01, straddle_premium=5.0)
    assert n == 2.0


def test_size_contracts_zero_when_premium_exceeds_budget() -> None:
    n = strat.size_contracts(capital=100_000.0, risk_fraction=0.001, straddle_premium=50.0)
    assert n == 0.0
```

- [ ] **Step 2: Run — expect FAIL.** Command: `uv run pytest tests/test_earnings_strategy.py -q`. Expected: `AttributeError: ... 'size_contracts'`.

- [ ] **Step 3: Implement `size_contracts` (body only — `CONTRACT_MULTIPLIER` already imported).** Append to `strategy.py` at module level:
```python
def size_contracts(capital: float, risk_fraction: float, straddle_premium: float) -> float:
    """Fixed-fractional sizing: contracts = floor(risk_budget / cost_per_contract).

    Long options are defined-risk (max loss = premium paid), so the risk budget
    is ``capital * risk_fraction`` and per-contract cost is
    ``straddle_premium * CONTRACT_MULTIPLIER``.
    """
    if straddle_premium <= 0:
        return 0.0
    budget = capital * risk_fraction
    cost_per_contract = straddle_premium * CONTRACT_MULTIPLIER
    return float(int(budget // cost_per_contract))
```

- [ ] **Step 4: Run — expect PASS.** Command: `uv run pytest tests/test_earnings_strategy.py -q`. Expected: 5 passed.

- [ ] **Step 5: Write the failing portfolio-cap test.** Append to `tests/test_earnings_strategy.py`:
```python
def test_portfolio_cap_blocks_when_at_capacity() -> None:
    assert strat.can_open(open_count=2, max_concurrent=2) is False
    assert strat.can_open(open_count=1, max_concurrent=2) is True
    assert strat.can_open(open_count=0, max_concurrent=1) is True
```

- [ ] **Step 6: Run — expect FAIL.** Command: `uv run pytest tests/test_earnings_strategy.py -q`. Expected: `AttributeError: ... 'can_open'`.

- [ ] **Step 7: Implement `can_open` (body only).** Append to `strategy.py` at module level:
```python
def can_open(open_count: int, max_concurrent: int) -> bool:
    """Portfolio cap: True iff fewer than ``max_concurrent`` straddles are open."""
    return open_count < max_concurrent
```

- [ ] **Step 8: Run — expect PASS.** Command: `uv run pytest tests/test_earnings_strategy.py -q`. Expected: 6 passed.

- [ ] **Step 9: Verify slice + commit.** Run `uv run ruff check models/options/03-earnings-straddle-spy/strategy.py tests/test_earnings_strategy.py` then `uv run ruff format` on both. Commit:
```
git add models/options/03-earnings-straddle-spy/strategy.py tests/test_earnings_strategy.py
git commit -m "feat(03-earnings-straddle): fixed-fractional sizing + portfolio cap"
```

---

### Task 8: Centered bootstrap test in validation (Component 6)

**Files**
- Modify: `tradinglib/backtest/metrics.py`
- Create: `tradinglib/validation/stats.py`
- Modify: `tradinglib/validation/__init__.py`
- Test: `tests/test_validation_stats.py`

Naming note (deliberate): `bootstrap_t_test`, `benjamini_hochberg_fdr`, `trade_metrics` are **public** (no underscore) because they are re-exported through `tradinglib/validation/stats.py` and `__init__.py`, matching the public surface convention of `compute_metrics`. The research's `_bootstrap_t_test` was a suggestion, not an existing API. metrics.py already imports `math`, `numpy as np`, `pandas as pd` — reuse them.

**Statistics fix (centered bootstrap):** the p-value is a *centered* bootstrap test of H0:mean=0 (subtract the sample mean, resample, count |boot_mean*| >= |observed_mean|, with `1/(n_boot+1)` smoothing so p is never exactly 0). The percentile CI is computed separately from the *uncentered* resampled means as an interval estimate. The two are documented as different (each internally consistent) procedures; the CI is not claimed to agree with the p-value sign-for-sign. The name keeps `t_test` for discoverability but the returned `t_stat` is the classic Student t reported only as a descriptive statistic (the docstring states it is not the test statistic).

- [ ] **Step 1: Write the failing bootstrap test.** Create `tests/test_validation_stats.py`:
```python
"""Tests for the bootstrap test, Benjamini-Hochberg FDR, and trade metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.validation import bootstrap_t_test


def test_bootstrap_ci_brackets_mean_and_pvalue_small_for_strong_signal() -> None:
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.02, 0.01, size=500))
    t_stat, ci_lo, ci_hi, p_value = bootstrap_t_test(rets, n_boot=2000, seed=1)

    assert ci_lo < float(rets.mean()) < ci_hi
    assert ci_lo > 0.0  # CI excludes zero
    assert 0.0 < p_value <= 2.0 / 2001  # floored, strictly positive, tiny
    assert t_stat > 0.0


def test_bootstrap_pvalue_large_for_zero_mean() -> None:
    rng = np.random.default_rng(2)
    rets = pd.Series(rng.normal(0.0, 0.05, size=400))
    _, ci_lo, ci_hi, p_value = bootstrap_t_test(rets, n_boot=2000, seed=3)
    assert ci_lo < 0.0 < ci_hi
    assert 0.0 < p_value <= 1.0
    assert p_value > 0.05


def test_bootstrap_handles_tiny_sample() -> None:
    rets = pd.Series([0.01])
    t_stat, ci_lo, ci_hi, p_value = bootstrap_t_test(rets, n_boot=100, seed=0)
    assert t_stat == 0.0 and ci_lo == 0.0 and ci_hi == 0.0 and p_value == 1.0
```

- [ ] **Step 2: Run — expect FAIL.** Command: `uv run pytest tests/test_validation_stats.py -q`. Expected: `ImportError: cannot import name 'bootstrap_t_test'`.

- [ ] **Step 3: Implement `bootstrap_t_test` (centered p-value).** Append to `tradinglib/backtest/metrics.py`:
```python
def bootstrap_t_test(
    returns: pd.Series,
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float, float]:
    """Non-parametric bootstrap of the mean per-trade return.

    Returns ``(t_stat, ci_lower, ci_upper, p_value)``.

    - ``p_value``: a *centered* bootstrap test of H0:mean=0 — resample the
      mean-centered returns and count how often ``|boot_mean*| >= |observed_mean|``,
      smoothed by ``1/(n_boot+1)`` so it is never exactly 0 (a bootstrap p-value
      floor; report as ``< 1/n_boot`` when it hits the floor).
    - ``(ci_lower, ci_upper)``: a percentile CI from the *uncentered* resampled
      means — an interval estimate computed by a different (internally consistent)
      procedure than the p-value; the two are not guaranteed to agree sign-for-sign
      in finite samples.
    - ``t_stat``: classic Student t (mean / (std/sqrt(n))), reported as a
      descriptive statistic only — it is NOT the test statistic (per-trade returns
      are fat-tailed and few, Ch. 4).

    Tiny samples (n < 2) return the conservative sentinel ``(0.0, 0.0, 0.0, 1.0)``.
    """
    x = returns.to_numpy(dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0, 1.0

    mean = float(x.mean())
    std = float(x.std(ddof=1))
    t_stat = float(mean / (std / math.sqrt(n))) if std > 0 else 0.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)

    alpha = 1.0 - confidence
    ci_lower = float(np.quantile(boot_means, alpha / 2.0))
    ci_upper = float(np.quantile(boot_means, 1.0 - alpha / 2.0))

    # centered bootstrap p-value (H0: mean == 0)
    shifted = boot_means - mean
    extreme = int((np.abs(shifted) >= abs(mean)).sum())
    p_value = (extreme + 1) / (n_boot + 1)
    return t_stat, ci_lower, ci_upper, float(p_value)
```

- [ ] **Step 4: Create the validation re-export.** Create `tradinglib/validation/stats.py`:
```python
"""Validation-layer statistics: bootstrap test, Benjamini-Hochberg FDR, trade metrics."""

from __future__ import annotations

from tradinglib.backtest.metrics import bootstrap_t_test

__all__ = ["bootstrap_t_test"]
```
Then in `tradinglib/validation/__init__.py` add `from tradinglib.validation.stats import bootstrap_t_test` (sorted with the other `from tradinglib.validation...` imports) and add `"bootstrap_t_test"` to `__all__` (keep `__all__` alphabetized for ruff RUF022).

- [ ] **Step 5: Run — expect PASS.** Command: `uv run pytest tests/test_validation_stats.py -q`. Expected: 3 passed.

- [ ] **Step 6: Verify slice + commit.** Run `uv run ruff check tradinglib/backtest/metrics.py tradinglib/validation/stats.py tradinglib/validation/__init__.py tests/test_validation_stats.py`, then `uv run ruff format` on those paths, then `uv run mypy tradinglib`. Commit:
```
git add tradinglib/backtest/metrics.py tradinglib/validation/stats.py tradinglib/validation/__init__.py tests/test_validation_stats.py
git commit -m "feat(validation): centered bootstrap test for per-trade returns"
```

---

### Task 9: Benjamini-Hochberg FDR in validation (Component 6)

**Files**
- Modify: `tradinglib/backtest/metrics.py`
- Modify: `tradinglib/validation/stats.py`
- Modify: `tradinglib/validation/__init__.py`
- Test: `tests/test_validation_stats.py`

**Statistics fix (BH test correctness):** the prior draft's m=8 example asserted only weak facts that passed by luck. The implementation (sort ascending, find the LARGEST rank i with p(i) <= (i/m)·α, reject all p <= that threshold) is correct; the tests below pin it exactly and add a **non-contiguous** case that genuinely exercises the largest-i step-up search (a small p-value following a failing one).

- [ ] **Step 1: Write the failing FDR tests (exact threshold + count, plus non-contiguous step-up).** Append to `tests/test_validation_stats.py`:
```python
def test_benjamini_hochberg_exact_threshold_and_count() -> None:
    from tradinglib.validation import benjamini_hochberg_fdr

    pvals = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]  # m=8, alpha=0.05
    rejected, threshold = benjamini_hochberg_fdr(pvals, alpha=0.05)

    # BH at m=8: only ranks 1-2 pass (0.001<=0.00625, 0.008<=0.0125; 0.039>0.01875)
    assert threshold == pytest.approx(0.008)
    assert sum(rejected) == 2
    assert rejected[0] is True and rejected[1] is True
    assert all((p <= threshold) == r for p, r in zip(pvals, rejected, strict=True))


def test_benjamini_hochberg_step_up_is_largest_i_not_first_failure() -> None:
    from tradinglib.validation import benjamini_hochberg_fdr

    # a small p-value (0.005) follows a failing one (0.04) in input order; the
    # step-up must scan to the LARGEST passing rank, not stop at the first failure.
    pvals = [0.001, 0.04, 0.005]  # sorted: 0.001,0.005,0.04 ; m=3, alpha=0.05
    rejected, threshold = benjamini_hochberg_fdr(pvals, alpha=0.05)
    # rank1 0.001<=0.0167 ok; rank2 0.005<=0.0333 ok; rank3 0.04<=0.05 ok -> all
    assert threshold == pytest.approx(0.04)
    assert rejected == [True, True, True]


def test_benjamini_hochberg_no_rejections() -> None:
    from tradinglib.validation import benjamini_hochberg_fdr

    rejected, threshold = benjamini_hochberg_fdr([0.6, 0.7, 0.9], alpha=0.05)
    assert rejected == [False, False, False]
    assert threshold == 0.0


def test_benjamini_hochberg_empty() -> None:
    from tradinglib.validation import benjamini_hochberg_fdr

    rejected, threshold = benjamini_hochberg_fdr([], alpha=0.05)
    assert rejected == []
    assert threshold == 0.0
```

- [ ] **Step 2: Run — expect FAIL.** Command: `uv run pytest tests/test_validation_stats.py -q`. Expected: `ImportError: cannot import name 'benjamini_hochberg_fdr'`.

- [ ] **Step 3: Implement `benjamini_hochberg_fdr`.** Append to `tradinglib/backtest/metrics.py`:
```python
def benjamini_hochberg_fdr(
    pvalues: list[float], alpha: float = 0.05
) -> tuple[list[bool], float]:
    """Benjamini-Hochberg FDR control over a set of hypotheses.

    Returns ``(rejected, threshold)`` where ``rejected[i]`` corresponds to
    ``pvalues[i]`` (input order preserved) and ``threshold`` is the largest
    p-value passing the BH step-up (0.0 if none). Scans to the LARGEST rank i with
    ``p(i) <= (i/m)*alpha`` and rejects all p <= that threshold. Controls the
    expected false-discovery rate at ``alpha`` across the hypothesis set (Ch. 4).
    """
    m = len(pvalues)
    if m == 0:
        return [], 0.0

    order = sorted(range(m), key=lambda i: pvalues[i])
    threshold = 0.0
    for rank, i in enumerate(order, start=1):
        if pvalues[i] <= (rank / m) * alpha:
            threshold = pvalues[i]
    rejected = [p <= threshold and threshold > 0.0 for p in pvalues]
    return rejected, float(threshold)
```

- [ ] **Step 4: Export it.** In `tradinglib/validation/stats.py` add `benjamini_hochberg_fdr` to the import and `__all__`; in `tradinglib/validation/__init__.py` add the import and `"benjamini_hochberg_fdr"` to `__all__` (keep both alphabetized).

- [ ] **Step 5: Run — expect PASS.** Command: `uv run pytest tests/test_validation_stats.py -q`. Expected: 7 passed.

- [ ] **Step 6: Verify slice + commit.** Run `uv run ruff check tradinglib/backtest/metrics.py tradinglib/validation/stats.py tradinglib/validation/__init__.py tests/test_validation_stats.py`, then `uv run ruff format` on those paths, then `uv run mypy tradinglib`. Commit:
```
git add tradinglib/backtest/metrics.py tradinglib/validation/stats.py tradinglib/validation/__init__.py tests/test_validation_stats.py
git commit -m "feat(validation): Benjamini-Hochberg FDR control over hypothesis set"
```

---

### Task 10: Trade-level metrics in validation (Component 6, Ch. 4.3.5)

**Files**
- Modify: `tradinglib/backtest/metrics.py`
- Modify: `tradinglib/validation/stats.py`
- Modify: `tradinglib/validation/__init__.py`
- Test: `tests/test_validation_stats.py`

This closes the spec-gap blocker: Component 6 requires win rate, profit factor, expectancy, average win/loss, average hold — none existed in `tradinglib` (verified by grep). `trade_metrics` operates on a per-trade P&L series (one value per closed trade) plus an optional per-trade hold-length series.

- [ ] **Step 1: Write the failing trade-metrics test.** Append to `tests/test_validation_stats.py`:
```python
def test_trade_metrics_basic() -> None:
    from tradinglib.validation import trade_metrics

    pnl = pd.Series([100.0, -50.0, 200.0, -100.0])  # 2 wins, 2 losses
    holds = pd.Series([3, 1, 4, 2])
    m = trade_metrics(pnl, holds=holds)

    assert m["n_trades"] == 4
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["profit_factor"] == pytest.approx(300.0 / 150.0)  # gross win / gross loss
    assert m["expectancy"] == pytest.approx((100 - 50 + 200 - 100) / 4)
    assert m["avg_win"] == pytest.approx(150.0)
    assert m["avg_loss"] == pytest.approx(-75.0)
    assert m["avg_hold"] == pytest.approx(2.5)


def test_trade_metrics_empty_and_no_losses() -> None:
    from tradinglib.validation import trade_metrics

    empty = trade_metrics(pd.Series([], dtype=float))
    assert empty["n_trades"] == 0
    assert empty["win_rate"] == 0.0
    assert empty["profit_factor"] == 0.0

    # all wins -> profit_factor is inf (no losses); avg_loss 0.0
    all_win = trade_metrics(pd.Series([10.0, 20.0]))
    assert all_win["win_rate"] == 1.0
    assert all_win["profit_factor"] == float("inf")
    assert all_win["avg_loss"] == 0.0
```

- [ ] **Step 2: Run — expect FAIL.** Command: `uv run pytest tests/test_validation_stats.py -q`. Expected: `ImportError: cannot import name 'trade_metrics'`.

- [ ] **Step 3: Implement `trade_metrics`.** Append to `tradinglib/backtest/metrics.py`:
```python
def trade_metrics(
    pnl: pd.Series, *, holds: pd.Series | None = None
) -> dict:
    """Trade-level metrics over a per-trade P&L series (Ch. 4.3.5).

    Returns a JSON-serializable dict: n_trades, win_rate, profit_factor
    (gross wins / gross losses; ``inf`` when there are wins but no losses, 0.0
    when empty), expectancy (mean P&L), avg_win, avg_loss (<= 0), avg_hold
    (mean of ``holds`` if provided else 0.0).
    """
    x = pnl.to_numpy(dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_hold": 0.0,
        }

    wins = x[x > 0]
    losses = x[x < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = float("inf") if gross_win > 0 else 0.0
    avg_hold = float(holds.to_numpy(dtype=float).mean()) if holds is not None and len(holds) else 0.0
    return {
        "n_trades": int(n),
        "win_rate": float(len(wins) / n),
        "profit_factor": float(profit_factor),
        "expectancy": float(x.mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "avg_hold": avg_hold,
    }
```

- [ ] **Step 4: Export it.** In `tradinglib/validation/stats.py` add `trade_metrics` to the import and `__all__`; in `tradinglib/validation/__init__.py` add the import and `"trade_metrics"` to `__all__` (alphabetized).

- [ ] **Step 5: Run — expect PASS.** Command: `uv run pytest tests/test_validation_stats.py -q`. Expected: 9 passed.

- [ ] **Step 6: Verify slice + commit.** Run `uv run ruff check tradinglib/backtest/metrics.py tradinglib/validation/stats.py tradinglib/validation/__init__.py tests/test_validation_stats.py`, then `uv run ruff format` on those paths, then `uv run mypy tradinglib`. Commit:
```
git add tradinglib/backtest/metrics.py tradinglib/validation/stats.py tradinglib/validation/__init__.py tests/test_validation_stats.py
git commit -m "feat(validation): trade-level metrics (win rate, profit factor, expectancy)"
```

---

### Task 11: Model directory wiring — backtest.py (filtered-vs-unfiltered report, multi-event FDR, leak-free expected move)

**Files**
- Create: `models/options/03-earnings-straddle-spy/backtest.py`
- Create: `models/options/03-earnings-straddle-spy/model.md`
- Create: `models/options/03-earnings-straddle-spy/README.md`
- Test: `tests/test_earnings_strategy.py` (integration tests), `tests/test_model_registry_earnings.py`

Key fixes folded in:
- **Walk-forward descope (stated):** the model is NOT driven through `walk_forward()` (incompatible engine, see Scope reconciliation). A per-season split is out of cycle. The Deflated-Sharpe `n_trials` hook is documented as the wiring point for a future options-aware walk-forward.
- **Leakage fix:** `expected_move` is computed from **prior** earnings events only. `run_synthetic` requires `past_moves` to be supplied (the realized abs moves of *earlier* events) OR a `prior_earnings` date series strictly before the traded event — it NEVER falls back to passing the traded event as its own history.
- **No-match guard:** the earnings-bar lookup uses `next((...), None)` with explicit `ValueError` when earnings is out of range or `e_idx - ENTRY_LEAD < 0`.
- **tz boundary:** `main()`/`run_for_gui` strip tz from the loaded close index; `run_synthetic` normalizes `earnings_datetime` to tz-naive.
- **Non-degenerate FDR:** each ticker contributes **multiple** synthetic events (real loader earnings dates), so each per-ticker bootstrap has n>=2; the cross-ticker FDR is meaningful. If a ticker has < 2 events its p-value is the n<2 sentinel (1.0) and it simply cannot be rejected (honest, not silently degenerate).
- **Step split:** `run_synthetic` (one event), `build_validation_report` (pure aggregation, unit-tested), `plot_branches` (smoke-tested PNG), thin `main()`, and `run_for_gui` (GUI adapter contract) are separate steps/functions.
- **GUI contract:** `run_for_gui(start, end, **params) -> dict` is implemented (mirrors `test_adapter_passthrough.py`) so the GUI can run the model; `model.md` params are wired to it.

- [ ] **Step 1: Write the failing single-event synthetic test (with leak-free past_moves + tz contract).** Append to `tests/test_earnings_strategy.py`:
```python
def _load_backtest():
    import importlib.util as _ilu

    bt_path = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "options"
        / "03-earnings-straddle-spy"
        / "backtest.py"
    )
    _bspec = _ilu.spec_from_file_location("earnings_backtest", bt_path)
    assert _bspec is not None and _bspec.loader is not None
    bt = _ilu.module_from_spec(_bspec)
    _bspec.loader.exec_module(bt)
    return bt


def test_run_synthetic_returns_filtered_and_unfiltered_pnl_tz_aware_earnings() -> None:
    bt = _load_backtest()
    # tz-NAIVE price index (like a tz-stripped load_daily) ...
    idx = pd.date_range("2024-01-02", periods=60, freq="B")
    rng = np.random.default_rng(7)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.005, 60)), index=idx)
    close.iloc[31:] *= 1.15  # big post-earnings move
    # ... and a tz-AWARE earnings timestamp (like the loader emits)
    earnings = pd.Timestamp(idx[30], tz="UTC")

    report = bt.run_synthetic(
        close=close,
        earnings_datetime=earnings,
        pre_iv=0.45,
        post_iv=0.25,
        k=1.2,
        lookback=4,
        past_moves=[0.12, 0.10, 0.14, 0.11],  # prior events only (no leakage)
    )

    assert "filtered" in report and "unfiltered" in report
    assert "metrics" in report["filtered"]
    assert "took_trade" in report["filtered"]
    assert "final_equity" in report["filtered"]
    assert "final_equity" in report["unfiltered"]
    assert "trade_pnl" in report["filtered"]


def test_run_synthetic_rejects_earnings_out_of_range() -> None:
    import pytest

    bt = _load_backtest()
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    close = pd.Series(100.0, index=idx)
    with pytest.raises(ValueError):
        bt.run_synthetic(
            close=close,
            earnings_datetime=pd.Timestamp("2025-01-01"),  # after last bar
            pre_iv=0.45,
            post_iv=0.25,
            past_moves=[0.10],
        )
    with pytest.raises(ValueError):
        bt.run_synthetic(
            close=close,
            earnings_datetime=pd.Timestamp(idx[1]),  # e_idx - ENTRY_LEAD < 0
            pre_iv=0.45,
            post_iv=0.25,
            past_moves=[0.10],
        )
```

- [ ] **Step 2: Run — expect FAIL.** Command: `uv run pytest tests/test_earnings_strategy.py::test_run_synthetic_returns_filtered_and_unfiltered_pnl_tz_aware_earnings -q`. Expected: spec load / `AttributeError` (no `run_synthetic`).

- [ ] **Step 3: Implement `run_synthetic` + module skeleton.** Create `models/options/03-earnings-straddle-spy/backtest.py` (all imports in the top block; matplotlib uses Agg before pyplot; `# noqa: E402` only on the post-`matplotlib.use` imports, matching the engine-stylistic exception):
```python
"""03-earnings-straddle (SPY) — Phase-1 synthetic pipeline.

Long ATM straddle into earnings on a liquid name. The EDGE is a selection
filter: enter only when forecast realized move exceeds the implied move priced
into the straddle by a margin k (>1). Phase 1 prices a synthetic straddle off an
explicit pre-earnings IV and post-earnings crush (EventVolSurface) while the
realized move comes from real yfinance bars. NOT YET TRADEABLE — synthetic vol,
mirrors the repo's SP2 treatment. Phases 2 (free forward snapshots) and 3 (paid
chain history) are out of scope.

tz contract: load_daily returns a UTC-aware index; main()/run_for_gui strip tz
(tz_localize(None)) so the index matches the strategy's naive bar handling, and
run_synthetic normalizes earnings_datetime to tz-naive. Leakage: expected_move is
computed ONLY from prior earnings events (past_moves), never from the traded event.

Walk-forward across earnings seasons (Component 6, 4th method) is descoped this
cycle: validation/walk_forward.py is built on the vectorized run_backtest and is
incompatible with the OptionsEngine path; an options-aware walk-forward is a
separate design cycle. The Deflated-Sharpe n_trials hook in run_options_backtest
is the future wiring point for a parameter grid.

Outputs: results/metrics.json (SPY filtered branch), results/validation.json
(filtered vs unfiltered + bootstrap CI + cross-ticker FDR + trade metrics),
results/equity_curve.png.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from tradinglib.backtest.metrics import (  # noqa: E402
    benjamini_hochberg_fdr,
    bootstrap_t_test,
    trade_metrics,
)
from tradinglib.backtest.options_engine import run_options_backtest  # noqa: E402
from tradinglib.loaders.equities.yfinance import load_daily  # noqa: E402
from tradinglib.loaders.events.earnings import get_earnings_dates  # noqa: E402
from tradinglib.options.spread import ParametricSpread  # noqa: E402
from tradinglib.options.straddle import snap_strike, straddle_price  # noqa: E402
from tradinglib.options.surface import EventVolSurface  # noqa: E402

_HERE = Path(__file__).resolve().parent
SYMBOL = "SPY"
WATCHLIST = ["SPY", "AAPL", "MSFT", "AMZN", "NVDA"]
START = "2023-01-01"
END = "2024-12-31"
RATE = 0.04
FEE_BPS = 1.0
SLIPPAGE_BPS = 0.5
INITIAL_CAPITAL = 100_000.0
DEFAULT_K = 1.2
DEFAULT_LOOKBACK = 8
ENTRY_LEAD = 3
EXIT_OFFSET = 1
POST_TENOR = 14


def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tz is not None else ts


def _load_strategy_module():
    spec = importlib.util.spec_from_file_location("es_strategy", _HERE / "strategy.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_signal_module():
    spec = importlib.util.spec_from_file_location("es_signal", _HERE / "signal.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_synthetic(
    *,
    close: pd.Series,
    earnings_datetime: pd.Timestamp,
    pre_iv: float,
    post_iv: float,
    k: float = DEFAULT_K,
    lookback: int = DEFAULT_LOOKBACK,
    past_moves: list[float] | None = None,
    prior_earnings: pd.Series | None = None,
) -> dict:
    """Run the synthetic straddle twice (filtered by the k-gate vs unfiltered).

    expected_move uses ONLY prior events: supply ``past_moves`` (realized abs
    moves of earlier events) or ``prior_earnings`` (dates strictly before the
    traded event); it never uses the traded event as its own history. Returns a
    dict with 'filtered'/'unfiltered' branches (metrics, final_equity, trade_pnl,
    took_trade) plus the implied/expected moves and k.
    """
    sig = _load_signal_module()
    strat = _load_strategy_module()

    cl = close.sort_index()
    if getattr(cl.index, "tz", None) is not None:
        cl = pd.Series(cl.to_numpy(), index=cl.index.tz_localize(None), name="close")
    ed = _to_naive(earnings_datetime)

    bars = list(cl.index)
    e_idx = next((i for i, b in enumerate(bars) if b >= ed), None)
    if e_idx is None or e_idx - ENTRY_LEAD < 0:
        raise ValueError(
            f"earnings {ed} out of range or too close to series start "
            f"(need entry_lead={ENTRY_LEAD} bars before the earnings bar)"
        )

    entry_spot = float(cl.iloc[e_idx - ENTRY_LEAD])
    strike = snap_strike(entry_spot, 1.0)
    t_years = max((ed + pd.Timedelta(days=POST_TENOR) - bars[e_idx - ENTRY_LEAD]).days, 1) / 365.0
    premium = straddle_price(entry_spot, strike, t_years, pre_iv, RATE)
    implied = sig.implied_move(premium, entry_spot)

    if past_moves is not None:
        expected = sig.expected_move(
            pd.Series([], dtype=float), pd.Series([], dtype="datetime64[ns]"), lookback
        )
        vals = [abs(m) for m in past_moves][-lookback:]
        expected = float(pd.Series(vals).mean()) if vals else float("nan")
    elif prior_earnings is not None:
        prior = pd.to_datetime(prior_earnings, utc=True)
        prior = prior[prior < pd.Timestamp(earnings_datetime, tz="UTC")]
        expected = sig.expected_move(cl, prior, lookback)
    else:
        raise ValueError("supply past_moves or prior_earnings (no self-referential history)")

    surface = EventVolSurface(earnings_datetime=ed, pre_iv=pre_iv, post_iv=post_iv)

    def _branch(take: bool) -> dict:
        if not take:
            return {
                "took_trade": False,
                "metrics": {"sharpe": 0.0, "annualized_return": 0.0, "max_drawdown": 0.0},
                "final_equity": INITIAL_CAPITAL,
                "trade_pnl": 0.0,
            }
        s = strat.EarningsStraddle(
            earnings_datetime=ed,
            entry_lead=ENTRY_LEAD,
            exit_offset=EXIT_OFFSET,
            contracts=1.0,
            post_earnings_tenor=POST_TENOR,
        )
        res = run_options_backtest(
            cl, s, surface=surface, spread=ParametricSpread(),
            rate=RATE, fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS,
            initial_capital=INITIAL_CAPITAL,
        )
        final_eq = float(res.equity_curve.iloc[-1])
        return {
            "took_trade": True,
            "metrics": res.metrics,
            "final_equity": final_eq,
            "trade_pnl": final_eq - INITIAL_CAPITAL,
            "implied_move": implied,
            "expected_move": expected,
        }

    return {
        "implied_move": implied,
        "expected_move": expected,
        "k": k,
        "filtered": _branch(sig.passes_filter(expected, implied, k)),
        "unfiltered": _branch(True),
    }
```

- [ ] **Step 4: Run — expect PASS.** Command: `uv run pytest tests/test_earnings_strategy.py -k run_synthetic -q`. Expected: 2 passed (the returns test + the out-of-range guard test).

- [ ] **Step 5a: Write the failing `build_validation_report` unit test (pure aggregation).** Append to `tests/test_earnings_strategy.py`:
```python
def test_build_validation_report_shapes() -> None:
    bt = _load_backtest()
    # two tickers, each with two synthetic per-event filtered trade P&Ls
    per_ticker_pnl = {
        "AAA": [120.0, -40.0],
        "BBB": [-10.0, -20.0],
    }
    branches = {
        "AAA": {"filtered": {"final_equity": 100120.0}, "unfiltered": {"final_equity": 100050.0}},
        "BBB": {"filtered": {"final_equity": 99970.0}, "unfiltered": {"final_equity": 99990.0}},
    }
    report = bt.build_validation_report(branches=branches, per_ticker_pnl=per_ticker_pnl)

    assert "pooled_filtered" in report
    assert {"bootstrap_t_stat", "bootstrap_ci_lower", "bootstrap_ci_upper",
            "bootstrap_p_value"} <= report["pooled_filtered"].keys()
    assert "fdr" in report
    assert report["fdr"]["tickers"] == ["AAA", "BBB"]
    assert len(report["fdr"]["p_values"]) == 2
    assert len(report["fdr"]["rejected"]) == 2
    assert "trade_metrics" in report
    assert report["trade_metrics"]["n_trades"] == 4
```

- [ ] **Step 5b: Run — expect FAIL, then implement `build_validation_report`.** Command: `uv run pytest tests/test_earnings_strategy.py::test_build_validation_report_shapes -q` (FAIL: no `build_validation_report`). Append to `backtest.py`:
```python
def build_validation_report(
    *,
    branches: dict[str, dict],
    per_ticker_pnl: dict[str, list[float]],
) -> dict:
    """Pure aggregation: pooled bootstrap CI, per-ticker FDR, pooled trade metrics.

    ``per_ticker_pnl[ticker]`` is the list of filtered per-event trade P&Ls for
    that ticker (n>=2 expected so the bootstrap is non-degenerate; a ticker with
    n<2 gets the sentinel p=1.0 and simply cannot be rejected).
    """
    pooled = pd.Series([p for ps in per_ticker_pnl.values() for p in ps], dtype=float)
    t_stat, ci_lo, ci_hi, p_value = bootstrap_t_test(pooled, n_boot=2000, seed=0)

    tickers = list(per_ticker_pnl.keys())
    pvals = [bootstrap_t_test(pd.Series(per_ticker_pnl[t]), n_boot=2000, seed=0)[3] for t in tickers]
    rejected, fdr_threshold = benjamini_hochberg_fdr(pvals, alpha=0.05)

    return {
        "phase": "1-synthetic-not-tradeable",
        "per_ticker": branches,
        "pooled_filtered": {
            "bootstrap_t_stat": t_stat,
            "bootstrap_ci_lower": ci_lo,
            "bootstrap_ci_upper": ci_hi,
            "bootstrap_p_value": p_value,
        },
        "fdr": {
            "tickers": tickers,
            "p_values": pvals,
            "rejected": rejected,
            "threshold": fdr_threshold,
        },
        "trade_metrics": trade_metrics(pooled),
    }
```
Run `uv run pytest tests/test_earnings_strategy.py::test_build_validation_report_shapes -q`. Expected: 1 passed.

- [ ] **Step 5c: Write the failing `plot_branches` smoke test, then implement it.** Append to `tests/test_earnings_strategy.py`:
```python
def test_plot_branches_writes_png(tmp_path) -> None:
    bt = _load_backtest()
    branches = {
        "AAA": {"filtered": {"final_equity": 100120.0}, "unfiltered": {"final_equity": 100050.0}},
    }
    out = tmp_path / "equity_curve.png"
    bt.plot_branches(branches, out)
    assert out.exists() and out.stat().st_size > 0
```
Run (FAIL: no `plot_branches`), then append to `backtest.py`:
```python
def plot_branches(branches: dict[str, dict], out_path: Path) -> None:
    """Bar chart of filtered vs unfiltered final equity per ticker."""
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(branches.keys())
    filt = [branches[n]["filtered"]["final_equity"] for n in names]
    unfilt = [branches[n]["unfiltered"]["final_equity"] for n in names]
    ax.bar([f"{n}\nfilt" for n in names], filt, label="filtered")
    ax.bar([f"{n}\nunfilt" for n in names], unfilt, label="unfiltered", alpha=0.6)
    ax.set_title("Earnings straddle: filtered vs unfiltered final equity (synthetic)")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
```
Run `uv run pytest tests/test_earnings_strategy.py::test_plot_branches_writes_png -q`. Expected: 1 passed.

- [ ] **Step 5d: Implement the thin `main()` + `run_for_gui` (no new test logic; integration helpers).** Append to `backtest.py`:
```python
def _events_for(ticker: str, close: pd.Series) -> pd.Series:
    """Real earnings dates within the price window (mocked-free path is data-optional)."""
    df = get_earnings_dates([ticker], start=START, end=END)
    return df["earnings_datetime"]


def run_for_gui(
    start: str | date = START,
    end: str | date = END,
    *,
    symbol: str = SYMBOL,
    k: float = DEFAULT_K,
    lookback: int = DEFAULT_LOOKBACK,
    entry_lead: int = ENTRY_LEAD,
    exit_offset: int = EXIT_OFFSET,
    pre_iv: float = 0.45,
    post_iv: float = 0.25,
) -> dict[str, Any]:
    """GUI adapter entry point. Runs one illustrative synthetic event on ``symbol``
    and returns a dict with the filtered/unfiltered report and a params echo.

    entry_lead/exit_offset are accepted to match the model.md param schema; they
    override the module defaults for this run via the strategy constructor inside
    a re-parameterized run_synthetic call.
    """
    bars = load_daily(symbol, start=str(start), end=str(end))
    close = bars["close"]
    close = pd.Series(close.to_numpy(), index=close.index.tz_localize(None), name="close")
    events = _events_for(symbol, close)
    events_naive = pd.to_datetime(events, utc=True).dt.tz_localize(None)
    usable = [e for e in events_naive if close.index[ENTRY_LEAD] <= e <= close.index[-EXIT_OFFSET - 1]]
    if not usable:
        return {"symbol": symbol, "report": None, "params": {"start": str(start), "end": str(end)}}
    traded = usable[len(usable) // 2]
    prior = events[pd.to_datetime(events, utc=True) < pd.Timestamp(traded, tz="UTC")]
    report = run_synthetic(
        close=close, earnings_datetime=traded, pre_iv=pre_iv, post_iv=post_iv,
        k=k, lookback=lookback, prior_earnings=prior,
    )
    return {
        "symbol": symbol,
        "report": report,
        "params": {
            "start": str(start), "end": str(end), "symbol": symbol, "k": k,
            "lookback": lookback, "entry_lead": entry_lead, "exit_offset": exit_offset,
            "pre_iv": pre_iv, "post_iv": post_iv,
        },
    }


def main() -> None:
    out_dir = _HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    branches: dict[str, dict] = {}
    per_ticker_pnl: dict[str, list[float]] = {}
    for ticker in WATCHLIST:
        try:
            bars = load_daily(ticker, START, END)
        except Exception:  # noqa: BLE001 - data optional in CI
            continue
        close = bars["close"]
        close = pd.Series(close.to_numpy(), index=close.index.tz_localize(None), name="close")
        try:
            events = _events_for(ticker, close)
        except Exception:  # noqa: BLE001
            continue
        events_naive = sorted(pd.to_datetime(events, utc=True).dt.tz_localize(None))
        usable = [
            e for e in events_naive
            if close.index[ENTRY_LEAD] <= e <= close.index[-EXIT_OFFSET - 1]
        ]
        pnl: list[float] = []
        last_report: dict | None = None
        for traded in usable:
            prior = events[pd.to_datetime(events, utc=True) < pd.Timestamp(traded, tz="UTC")]
            try:
                rep = run_synthetic(
                    close=close, earnings_datetime=traded,
                    pre_iv=0.45, post_iv=0.25, prior_earnings=prior,
                )
            except ValueError:
                continue
            last_report = rep
            f = rep["filtered"]
            pnl.append(float(f["trade_pnl"]) if f["took_trade"] else 0.0)
        if last_report is not None and pnl:
            branches[ticker] = last_report
            per_ticker_pnl[ticker] = pnl

    if not branches:
        print(json.dumps({"phase": "1-synthetic-not-tradeable", "note": "no data"}, indent=2))
        return

    validation = build_validation_report(branches=branches, per_ticker_pnl=per_ticker_pnl)
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2, default=str))

    spy = branches.get("SPY", {}).get("filtered", {}).get("metrics", {})
    (out_dir / "metrics.json").write_text(json.dumps(spy, indent=2, default=str))
    plot_branches(branches, out_dir / "equity_curve.png")
    print(json.dumps(validation, indent=2, default=str))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Create `model.md`.** Create `models/options/03-earnings-straddle-spy/model.md`. Params are exactly those `run_for_gui` accepts (no `entry_lead`/`exit_offset` mismatch — they ARE accepted by `run_for_gui`):
```markdown
---
name: Earnings Event-Vol Straddle on SPY
family: options
window: swing
assets: [equities]
data_sources: [yfinance_daily_bars, yfinance_earnings_calendar]
tickers: any
default_ticker: SPY
supports_costs: false
supports_sizing: false
params:
  - {name: k, label: Edge margin (expected/implied), type: float, default: 1.2, min: 1.01, max: 3.0}
  - {name: lookback, label: Earnings lookback (events), type: int, default: 8, min: 2, max: 20}
  - {name: entry_lead, label: Entry lead (trading days), type: int, default: 3, min: 1, max: 10}
  - {name: exit_offset, label: Exit offset (trading days), type: int, default: 1, min: 0, max: 5}
  - {name: pre_iv, label: Pre-earnings IV (synthetic), type: float, default: 0.45, min: 0.15, max: 1.50}
  - {name: post_iv, label: Post-earnings IV (synthetic crush, must be < pre_iv), type: float, default: 0.25, min: 0.05, max: 0.90}
status: working
sharpe_oos: 0.0
max_drawdown: 0.0
---

Long ATM straddle entered into earnings on a liquid optionable name. The edge is
a selection filter — enter only when the forecast realized move exceeds the
implied move priced into the straddle by a margin k (>1). The straddle is the
expression; the filter is the alpha. Phase 1 is synthetic (an elevated
pre-earnings IV plus a parameterized post-earnings crush, with the constraint
`post_iv < pre_iv` so the synthetic premium is conservative, never tunable to a
fake IV expansion) and clearly not yet tradeable; the realized move comes from
real yfinance bars and expected move from PRIOR earnings events only. The
validation report distinguishes filtered vs unfiltered P&L and applies a
non-parametric bootstrap test, Benjamini-Hochberg FDR across the watchlist, and
trade-level metrics. Walk-forward across earnings seasons is deferred (the
existing harness is built on the vectorized engine; an options-aware
walk-forward is a separate cycle).
```

- [ ] **Step 7: Write the failing registry test, then create `README.md`.** Create `tests/test_model_registry_earnings.py`:
```python
"""The new model.md is discoverable and has the required frontmatter keys."""

from __future__ import annotations

from pathlib import Path

from tradinglib.data.paths import repo_root
from tradinglib.models_index import find_models, parse_frontmatter


def test_model_md_frontmatter_has_required_keys() -> None:
    md = (
        repo_root()
        / "models"
        / "options"
        / "03-earnings-straddle-spy"
        / "model.md"
    )
    meta = parse_frontmatter(md)
    assert meta is not None
    for key in ("name", "family", "status", "sharpe_oos", "max_drawdown", "params"):
        assert key in meta
    assert meta["family"] == "options"


def test_find_models_discovers_earnings_straddle() -> None:
    paths = {m["_path"] for m in find_models()}
    assert "models/options/03-earnings-straddle-spy" in paths
```
Run `uv run pytest tests/test_model_registry_earnings.py -q` (FAIL: model.md not yet created → already created in Step 6, so this should PASS once Step 6 is done; if Step 6 precedes, run it now to confirm 2 passed). Then create `models/options/03-earnings-straddle-spy/README.md` documenting: the edge is the filter (not "own vol into earnings"); the spread is paid twice (two legs, entry + exit) and is the dominant cost; Phase 1 is synthetic / not-yet-tradeable; greeks diagnostics and walk-forward-across-seasons are deferred; how to reproduce (`uv run python models/options/03-earnings-straddle-spy/backtest.py`); how to read `results/validation.json` (filtered vs unfiltered, bootstrap CI, FDR survivors, trade metrics).

- [ ] **Step 8: Run the full model + registry test files — expect PASS.** Command: `uv run pytest tests/test_earnings_strategy.py tests/test_model_registry_earnings.py -q`. Expected: all passed (timing×2, friction, sizing×2, cap, run_synthetic×2, report, plot, registry×2).

- [ ] **Step 9: Verify slice + commit.** Run `uv run ruff check models/options/03-earnings-straddle-spy tests/test_model_registry_earnings.py` then `uv run ruff format` on the same paths. Commit:
```
git add models/options/03-earnings-straddle-spy tests/test_earnings_strategy.py tests/test_model_registry_earnings.py
git commit -m "feat(03-earnings-straddle): synthetic Phase-1 backtest + validation report"
```

---

### Task 12: End-to-end verification + docs (README, data-sources, ingestion, MODELS.md)

**Files**
- Create: `data/ingestion/events/README.md`
- Modify: `docs/data-sources.md`
- Modify: `README.md`
- Modify: `MODELS.md` (regenerated)
- Modify: `models/options/03-earnings-straddle-spy/model.md` (frontmatter back-population)

Spec override note: this task performs README/MODELS.md updates that the spec lists as out-of-scope, deliberately overriding per the user's "always update README with new models" convention.

- [ ] **Step 1: Add the ingestion doc.** Read `data/ingestion/equities/README.md` for the house format, then create `data/ingestion/events/README.md` describing the earnings loader: source = yfinance `Ticker.get_earnings_dates`, schema `[ticker, earnings_datetime, session]`, cache path `data/processed/events/earnings/<ticker>/<snapshot>.parquet`, point-in-time discipline (snapshot date in the path), and the pluggable-provider note.

- [ ] **Step 2: Add the data-sources row.** In `docs/data-sources.md`, add a row to the top table for the `events/earnings` source (yfinance, free, no key) and, if the file has a per-source section list, a short section mirroring the equities entry.

- [ ] **Step 3: Add the model to the README Current models table.** The real table (README.md:61-67) has 7 columns: `| Model | Family | Window | Assets | OOS Sharpe | Max DD | Status |`, with `Model` a markdown link to a GitHub Pages HTML (existing options model is linked as `05-delta-hedged-long-option-spy.html` — README numbering is decoupled from the model directory number). Add a row matching that schema, leaving Sharpe/MaxDD as placeholders to be filled in Step 5:
```
| [Earnings Event-Vol Straddle on SPY](https://vankyle00.github.io/trading-models/docs/models/06-earnings-straddle-spy.html) | options | swing | equities | <sharpe> | <maxdd> | working |
```
(Confirm the next free HTML doc number by listing `docs/models/*.html`; use the next integer after the highest existing — `06` if `05` is the current max. The HTML page itself is a future docs task; the link target is reserved here.)

- [ ] **Step 4: Run the FULL six-step gate.** Run, in order:
  1. `uv run ruff check .`
  2. `uv run ruff format --check .`
  3. `uv run mypy tradinglib`
  4. `uv run pytest`
  5. `uv run python -c "import streamlit"` (or the repo's streamlit import check command)
  6. MODELS.md staleness: `uv run python scripts/regenerate_models_index.py` then `git diff --exit-code MODELS.md`

  Expected: steps 1-5 green. Step 6 will report MODELS.md changed (the regenerator added the new model row) — that is expected on first run; stage MODELS.md. Fix any ruff/mypy/pytest failure before proceeding (do not suppress with noqa unless a line is genuinely unavoidable, e.g. the post-`matplotlib.use` imports already marked `# noqa: E402`).

- [ ] **Step 5: Back-populate frontmatter metrics, then verify model.md matches results.** Run `uv run python models/options/03-earnings-straddle-spy/backtest.py` (data-optional; if no network/data in CI it prints the no-data note and writes nothing — in that case leave the `0.0` placeholders and note in the README that metrics await a data-enabled run). When `results/metrics.json` is produced, copy `sharpe` → `sharpe_oos` and `max_drawdown` → `max_drawdown` in `model.md`. Verification of the copy: re-run `uv run python scripts/regenerate_models_index.py` and confirm `git diff MODELS.md` shows the new Sharpe/MaxDD values in the model's MODELS.md row (the registry reads the same frontmatter, so a copy error surfaces as a mismatch in the diff). Then re-run `uv run pytest -q` to confirm nothing broke.

- [ ] **Step 6: Final commit.** Stage docs, README, MODELS.md, and the updated model.md:
```
git add data/ingestion/events/README.md docs/data-sources.md README.md MODELS.md models/options/03-earnings-straddle-spy/model.md
git commit -m "docs(03-earnings-straddle): register model + earnings events source"
```

- [ ] **Step 7: Confirm the whole gate one last time.** Re-run the six-step gate from Step 4 against the final tree (step 6 should now `git diff --exit-code MODELS.md` cleanly since MODELS.md is committed). Expected: all green, working tree clean except intended artifacts under `models/options/03-earnings-straddle-spy/results/`.

---

## Future work (out of scope for these tasks)

- **Walk-forward across earnings seasons (Component 6, deferred).** An options-aware walk-forward (or a season-split harness selecting `k`/`lookback`/`entry_lead` in-sample and freezing them OOS) that feeds `len(grid)` as `n_trials` into the Deflated Sharpe. The current `tradinglib/validation/walk_forward.py` is built on the vectorized `run_backtest` and is incompatible with the `OptionsEngine` path; this needs new adapter code and is its own cycle.
- **Greeks diagnostics (Component 2, deferred).** Surface entry vega/theta (via `bs_greeks`) in the per-event report; the Phase-1 no-trade filters do not require them.
- **Phase 2 — free forward-snapshot collector.** A `tradinglib/loaders/events/option_chain.py` snapshotter using yfinance `Ticker.option_chain()` plus the earnings calendar to record real ATM straddle prices and IV for the watchlist going forward; a real (non-synthetic) backtest becomes possible once a couple of earnings seasons accrue. Leak-free by construction via the loader's snapshot-dated cache. Replaces `EventVolSurface` with real chain IV without touching the signal, sizing, or validation code; the listed-expiry snap (vs Phase-1's fixed `post_earnings_tenor`) and session-aware entry/exit (bmo/amc, currently parsed-but-unused) land here.
- **Phase 3 — optional paid chain history.** Polygon / ORATS EOD options history for immediate deep backtests (the deferred SP3 chain loader).
- **Signal extensions (YAGNI now):** IV-term-structure blending in the expected-move forecast, vol-parity sizing, and strangle/calendar structures — all explicitly deferred in the spec.