# Earnings Straddle Phase 2 (Real Chain Data) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the synthetic IV path of `models/options/03-earnings-straddle-spy` with real option-chain data: a DoltHub historical chain loader, a yfinance forward-snapshot collector, and a quote-to-quote real-chain event backtest through the existing bootstrap/FDR harness.

**Architecture:** Two new loaders under `tradinglib/loaders/options/` share one canonical chain schema. Event mechanics (expiry/strike selection, quote-to-quote P&L, skip reasons) live in a new importable module in the model directory; a thin script runs the full universe and writes `results/real_chain_backtest.json`. No pricing model anywhere in the real-chain P&L.

**Tech Stack:** Python 3.12, pandas, httpx (DoltHub SQL API), yfinance (snapshots), pytest with all network mocked. Spec: `docs/specs/2026-06-09-earnings-straddle-real-chain-design.md`.

**Conventions you must follow (this repo):**
- Run everything via `uv run ...` from the repo root (`C:\Users\Administrator\trading-models`).
- Network libraries are NEVER called live in tests — mock `httpx` / `yf.Ticker` exactly like `tests/test_earnings_loader.py` does.
- Loader caches go under `data/processed/...` via `tradinglib.data.paths.processed_dir` — tests monkeypatch `processed_dir` to `tmp_path`.
- Lint: `uv run ruff check .` must stay clean (line length 100); types: repo runs mypy non-strict.
- The model directory is NOT a package — model modules are loaded via `importlib.util.spec_from_file_location` (see `backtest.py:74-87`).

**One deliberate deviation from the spec** (documented in Task 6): `signal.tradeable_event` is NOT used for chain tradeability, because it returns False when `expected` is NaN (no prior earnings history), which would silently drop forecast-less events from the *unfiltered* baseline. Instead `real_chain.run_event` checks the spread cap directly and uses `signal.passes_filter` (which already maps NaN → gate-does-not-fire) so the unfiltered branch keeps every tradeable event, matching the Phase-1 framing.

---

## File Structure

```
tradinglib/loaders/options/__init__.py        (new, empty — subpackage marker)
tradinglib/loaders/options/dolthub.py         (new — historical chain loader, Task 1)
tradinglib/loaders/options/yf_chain.py        (new — forward snapshot collector, Task 2)
scripts/collect_chain_snapshots.py            (new — daily snapshot runner, Task 2)
models/options/03-earnings-straddle-spy/real_chain.py   (new — event mechanics, Task 3)
scripts/earnings_straddle_real_chain_backtest.py         (new — universe runner, Task 4)
tests/test_dolthub_loader.py                  (new, Task 1)
tests/test_yf_chain_snapshot.py               (new, Task 2)
tests/test_earnings_real_chain.py             (new, Task 3)
models/options/03-earnings-straddle-spy/results/real_chain_backtest.json  (generated, Task 5)
model.md / README.md / docs/data-sources.md / spec amendment              (Task 6)
```

Canonical chain schema (both loaders): columns `[date, ticker, expiration, strike, right, bid, ask, iv]`; `right ∈ {"call", "put"}`; `date`/`expiration` tz-naive datetimes; numerics float. The yfinance snapshots add a `spot` float column.

---

### Task 1: DoltHub chain loader

**Files:**
- Create: `tradinglib/loaders/options/__init__.py` (empty)
- Create: `tradinglib/loaders/options/dolthub.py`
- Test: `tests/test_dolthub_loader.py`

Background: the DoltHub SQL API serves `post-no-preference/options`, table
`option_chain(date, act_symbol, expiration, strike, call_put, bid, ask, vol, delta, gamma, theta, vega, rho)`,
PK `(date, act_symbol, expiration, strike, call_put)`. Decimals arrive as JSON **strings**.
Every query MUST filter on exact `date` AND `act_symbol` (PK prefix) — anything else scans
~1e9 rows and times out (verified live 2026-06-09). Success responses carry
`"query_execution_status": "Success"` (or `"RowLimit"` when truncated).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dolthub_loader.py`:

```python
"""Tests for the DoltHub historical option-chain loader (httpx mocked, never live)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

_FAKE_ROWS = [
    {
        "date": "2026-06-05", "act_symbol": "AAPL", "expiration": "2026-06-18",
        "strike": "290.00", "call_put": "Call", "bid": "7.10", "ask": "7.30", "vol": "0.2935",
    },
    {
        "date": "2026-06-05", "act_symbol": "AAPL", "expiration": "2026-06-18",
        "strike": "290.00", "call_put": "Put", "bid": "6.80", "ask": "7.00", "vol": "0.2712",
    },
]

_COLUMNS = ["date", "ticker", "expiration", "strike", "right", "bid", "ask", "iv"]


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def _ok(rows: list[dict]) -> _FakeResponse:
    return _FakeResponse({"query_execution_status": "Success", "rows": rows})


def test_canonicalize_schema_and_coercion() -> None:
    from tradinglib.loaders.options import dolthub

    out = dolthub._canonicalize(_FAKE_ROWS, "AAPL")

    assert list(out.columns) == _COLUMNS
    assert set(out["right"]) == {"call", "put"}
    assert out["strike"].dtype == "float64"
    assert out["bid"].iloc[0] == pytest.approx(7.10) or out["bid"].iloc[0] == pytest.approx(6.80)
    assert pd.api.types.is_datetime64_any_dtype(out["expiration"])
    assert out["expiration"].dt.tz is None  # tz-naive by contract


def test_canonicalize_empty_returns_canonical_empty() -> None:
    from tradinglib.loaders.options import dolthub

    out = dolthub._canonicalize([], "AAPL")
    assert list(out.columns) == _COLUMNS
    assert out.empty


def test_load_chain_caches_and_skips_network_on_second_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.options import dolthub

    monkeypatch.setattr(dolthub, "processed_dir", lambda source: tmp_path / source)
    monkeypatch.setattr(dolthub.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fake_get(url: str, params: dict, timeout: float) -> _FakeResponse:
        calls["n"] += 1
        assert "date" in params["q"] and "act_symbol" in params["q"]  # PK-prefix discipline
        return _ok(_FAKE_ROWS)

    monkeypatch.setattr(dolthub.httpx, "get", fake_get)

    df1 = dolthub.load_chain("AAPL", "2026-06-05")
    df2 = dolthub.load_chain("AAPL", "2026-06-05")

    assert calls["n"] == 1  # second call served from parquet cache
    assert len(df1) == 2 and len(df2) == 2
    cached = list((tmp_path / "options" / "dolthub" / "AAPL").glob("*.parquet"))
    assert len(cached) == 1


def test_load_chain_empty_result_is_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.options import dolthub

    monkeypatch.setattr(dolthub, "processed_dir", lambda source: tmp_path / source)
    monkeypatch.setattr(dolthub.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fake_get(url: str, params: dict, timeout: float) -> _FakeResponse:
        calls["n"] += 1
        return _ok([])

    monkeypatch.setattr(dolthub.httpx, "get", fake_get)

    df1 = dolthub.load_chain("ZZZZ", "2026-06-05")
    df2 = dolthub.load_chain("ZZZZ", "2026-06-05")

    assert df1.empty and df2.empty
    assert calls["n"] == 1  # the miss is cached too


def test_query_error_status_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.options import dolthub

    monkeypatch.setattr(dolthub, "processed_dir", lambda source: tmp_path / source)
    monkeypatch.setattr(dolthub.time, "sleep", lambda s: None)

    def fake_get(url: str, params: dict, timeout: float) -> _FakeResponse:
        return _FakeResponse(
            {"query_execution_status": "Error", "query_execution_message": "context deadline exceeded"}
        )

    monkeypatch.setattr(dolthub.httpx, "get", fake_get)

    with pytest.raises(RuntimeError, match="context deadline exceeded"):
        dolthub.load_chain("AAPL", "2026-06-06")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dolthub_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tradinglib.loaders.options'`

- [ ] **Step 3: Write the loader**

Create empty `tradinglib/loaders/options/__init__.py`, then `tradinglib/loaders/options/dolthub.py`:

```python
"""DoltHub historical option-chain loader (post-no-preference/options).

Free SQL API over the community options database. Upstream schema:
``option_chain(date, act_symbol, expiration, strike, call_put, bid, ask, vol,
delta, gamma, theta, vega, rho)`` with PK ``(date, act_symbol, expiration,
strike, call_put)``. QUERY DISCIPLINE: every query must filter on exact
``date`` AND ``act_symbol`` (the PK prefix) — anything else scans a ~1e9-row
table and times out ("context deadline exceeded", observed 2026-06-09). The
dataset carries only 3-4 expirations per (date, symbol) — select weeklies and
monthlies roughly 2-7 weeks out — so consumers pick the nearest *available*
expiry, not an arbitrary listed one.

Canonical schema: ``[date, ticker, expiration, strike, right, bid, ask, iv]``
with ``right in {"call", "put"}``; decimals arrive as JSON strings and are
coerced to float. Cached to
``data/processed/options/dolthub/<ticker>/<date>.parquet``; an empty API
result caches an empty frame so the miss is remembered. httpx is mocked in
tests and never called live (repo convention).
"""

from __future__ import annotations

import time
from datetime import date, datetime

import httpx
import pandas as pd

from tradinglib.data.paths import processed_dir

SOURCE = "options"
_SUBDIR = "dolthub"
_API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
_OK_STATUSES = ("Success", "RowLimit")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series([], dtype="datetime64[ns]"),
            "ticker": pd.Series([], dtype="object"),
            "expiration": pd.Series([], dtype="datetime64[ns]"),
            "strike": pd.Series([], dtype="float64"),
            "right": pd.Series([], dtype="object"),
            "bid": pd.Series([], dtype="float64"),
            "ask": pd.Series([], dtype="float64"),
            "iv": pd.Series([], dtype="float64"),
        }
    )


def _canonicalize(rows: list[dict], ticker: str) -> pd.DataFrame:
    """API JSON rows -> canonical frame (strings coerced, Call/Put lowered)."""
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "ticker": ticker,
            "expiration": pd.to_datetime(df["expiration"]),
            "strike": df["strike"].astype(float),
            "right": df["call_put"].str.lower(),
            "bid": pd.to_numeric(df["bid"], errors="coerce"),
            "ask": pd.to_numeric(df["ask"], errors="coerce"),
            "iv": pd.to_numeric(df["vol"], errors="coerce"),
        }
    )
    return out.sort_values(["expiration", "strike", "right"]).reset_index(drop=True)


def _query(sql: str) -> list[dict]:
    """One SQL query against the DoltHub API; one retry on transient failure."""
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            resp = httpx.get(_API, params={"q": sql}, timeout=60.0)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("query_execution_status") not in _OK_STATUSES:
                raise RuntimeError(
                    f"DoltHub query failed: {payload.get('query_execution_message', 'unknown')}"
                )
            return payload.get("rows", [])
        except (httpx.HTTPError, RuntimeError) as err:
            last_err = err
            if attempt == 0:
                time.sleep(1.0)
    raise RuntimeError(f"DoltHub query failed after retry: {last_err}") from last_err


def load_chain(ticker: str, when: str | date | datetime, *, refresh: bool = False) -> pd.DataFrame:
    """Full available chain for one (ticker, trading date), cache-first."""
    d = pd.Timestamp(when).strftime("%Y-%m-%d")
    out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{d}.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    sql = (
        "SELECT `date`, expiration, strike, call_put, bid, ask, vol "
        f"FROM option_chain WHERE `date`='{d}' AND act_symbol='{ticker}'"
    )
    df = _canonicalize(_query(sql), ticker)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    time.sleep(0.5)  # politeness between live API calls
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dolthub_loader.py -v`
Expected: 5 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check tradinglib/loaders/options tests/test_dolthub_loader.py
git add tradinglib/loaders/options tests/test_dolthub_loader.py
git commit -m "feat: DoltHub historical option-chain loader (canonical schema, parquet cache)"
```

---

### Task 2: yfinance forward-snapshot collector

**Files:**
- Create: `tradinglib/loaders/options/yf_chain.py`
- Create: `scripts/collect_chain_snapshots.py`
- Test: `tests/test_yf_chain_snapshot.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_yf_chain_snapshot.py`:

```python
"""Tests for the yfinance forward chain-snapshot collector (yfinance mocked, never live)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

_COLUMNS = ["date", "ticker", "expiration", "strike", "right", "bid", "ask", "iv", "spot"]


def _fake_leg(strikes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strike": strikes,
            "bid": [1.0] * len(strikes),
            "ask": [1.2] * len(strikes),
            "impliedVolatility": [0.30] * len(strikes),
        }
    )


class _FakeTicker:
    """Two expirations: one inside the 45-day window, one far beyond it."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
        self.options = (
            (today + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
            (today + pd.Timedelta(days=200)).strftime("%Y-%m-%d"),
        )
        self.fast_info = {"lastPrice": 100.0}

    def option_chain(self, expiration: str) -> SimpleNamespace:
        return SimpleNamespace(calls=_fake_leg([95.0, 100.0]), puts=_fake_leg([95.0, 100.0]))


def test_snapshot_writes_canonical_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.options import yf_chain

    monkeypatch.setattr(yf_chain, "processed_dir", lambda source: tmp_path / source)
    monkeypatch.setattr(yf_chain.yf, "Ticker", _FakeTicker)

    counts = yf_chain.snapshot_chains(["AAPL"])

    assert counts["AAPL"] == 4  # 1 in-window expiration x 2 strikes x 2 rights
    files = list((tmp_path / "options" / "yf_snapshots" / "AAPL").glob("*.parquet"))
    assert len(files) == 1
    df = pd.read_parquet(files[0])
    assert list(df.columns) == _COLUMNS
    assert set(df["right"]) == {"call", "put"}
    assert (df["spot"] == 100.0).all()
    # the 200-day expiration is beyond the 45-day cutoff and must be absent
    assert df["expiration"].nunique() == 1


def test_snapshot_is_idempotent_per_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.options import yf_chain

    monkeypatch.setattr(yf_chain, "processed_dir", lambda source: tmp_path / source)
    monkeypatch.setattr(yf_chain.yf, "Ticker", _FakeTicker)

    first = yf_chain.snapshot_chains(["AAPL"])
    second = yf_chain.snapshot_chains(["AAPL"])

    assert first["AAPL"] == 4
    assert second["AAPL"] == -1  # sentinel: already snapshotted today, no refetch
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_yf_chain_snapshot.py -v`
Expected: FAIL with `ImportError: cannot import name 'yf_chain'`

- [ ] **Step 3: Write the collector**

Create `tradinglib/loaders/options/yf_chain.py`:

```python
"""Forward option-chain snapshot collector (yfinance).

Same canonical schema as the DoltHub loader (``[date, ticker, expiration,
strike, right, bid, ask, iv]``) plus a ``spot`` column, so the backtest can
consume either source once enough forward history accrues. Snapshots are
point-in-time by definition: one parquet per (ticker, snapshot date) at
``data/processed/options/yf_snapshots/<ticker>/<date>.parquet``, and an
existing file for today is never re-fetched (idempotent — a snapshot has no
meaningful "refresh"). Only expirations within ``MAX_DAYS`` calendar days are
collected (enough to cover earnings straddle tenors). yfinance is mocked in
tests and never called live (repo convention).
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from tradinglib.data.paths import processed_dir

SOURCE = "options"
_SUBDIR = "yf_snapshots"
MAX_DAYS = 45

_COLUMNS = ["date", "ticker", "expiration", "strike", "right", "bid", "ask", "iv", "spot"]


def _canonicalize_expiry(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    *,
    ticker: str,
    snapshot: str,
    expiration: str,
    spot: float,
) -> pd.DataFrame:
    frames = []
    for right, leg in (("call", calls), ("put", puts)):
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.Timestamp(snapshot),
                    "ticker": ticker,
                    "expiration": pd.Timestamp(expiration),
                    "strike": leg["strike"].astype(float),
                    "right": right,
                    "bid": pd.to_numeric(leg["bid"], errors="coerce"),
                    "ask": pd.to_numeric(leg["ask"], errors="coerce"),
                    "iv": pd.to_numeric(leg["impliedVolatility"], errors="coerce"),
                    "spot": float(spot),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["expiration", "strike", "right"]).reset_index(drop=True)


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series([], dtype="object") for c in _COLUMNS})


def snapshot_chains(tickers: list[str], *, max_days: int = MAX_DAYS) -> dict[str, int]:
    """Snapshot near-term chains; returns {ticker: rows written} (-1 = already done today)."""
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    cutoff = pd.Timestamp(snapshot) + pd.Timedelta(days=max_days)
    counts: dict[str, int] = {}
    for ticker in tickers:
        out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{snapshot}.parquet"
        if out.exists():
            counts[ticker] = -1
            continue
        t = yf.Ticker(ticker)
        spot = float(t.fast_info["lastPrice"])
        frames = []
        for exp in t.options:
            if pd.Timestamp(exp) > cutoff:
                continue
            chain = t.option_chain(exp)  # fetch once per expiration
            frames.append(
                _canonicalize_expiry(
                    chain.calls,
                    chain.puts,
                    ticker=ticker,
                    snapshot=snapshot,
                    expiration=exp,
                    spot=spot,
                )
            )
        df = pd.concat(frames, ignore_index=True) if frames else _empty()
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        counts[ticker] = len(df)
    return counts
```

Create `scripts/collect_chain_snapshots.py`:

```python
"""Daily forward chain snapshot for the earnings-straddle watchlist (Phase 2).

Run once per trading day (Windows Task Scheduler / cron — see
docs/data-sources.md). Idempotent: a ticker already snapshotted today is
skipped. Value accrues with calendar time; this forward dataset is the only
true point-in-time OOS source for the earnings straddle.

    uv run python scripts/collect_chain_snapshots.py
"""

from __future__ import annotations

from tradinglib.loaders.options.yf_chain import snapshot_chains

WATCHLIST = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "NFLX", "AMD"]


def main() -> None:
    for ticker, n in snapshot_chains(WATCHLIST).items():
        status = "already snapshotted today" if n < 0 else f"{n} rows"
        print(f"{ticker}: {status}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_yf_chain_snapshot.py -v`
Expected: 2 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check tradinglib/loaders/options scripts/collect_chain_snapshots.py tests/test_yf_chain_snapshot.py
git add tradinglib/loaders/options/yf_chain.py scripts/collect_chain_snapshots.py tests/test_yf_chain_snapshot.py
git commit -m "feat: yfinance forward chain-snapshot collector + daily watchlist script"
```

---

### Task 3: Real-chain event mechanics (`real_chain.py`)

**Files:**
- Create: `models/options/03-earnings-straddle-spy/real_chain.py`
- Test: `tests/test_earnings_real_chain.py`

This is the heart of the cycle. Quote-to-quote: entry buys ATM call+put at the
real ask, exit sells both at the real bid. P&L math (verify by hand): with
`fee_bps=1.0`, entry cost 9.00, exit value 5.00 →
`fees = (9.00 + 5.00) × 100 × 0.0001 = 0.14`, `pnl = (5.00 − 9.00) × 100 − 0.14 = −400.14`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_earnings_real_chain.py`:

```python
"""Tests for the Phase-2 quote-to-quote real-chain event mechanics."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "options" / "03-earnings-straddle-spy"
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"es_{name}", _MODEL_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load("real_chain")
sig = _load("signal")


def _chain(rows: list[tuple]) -> pd.DataFrame:
    """rows: (expiration, strike, right, bid, ask)"""
    return pd.DataFrame(
        {
            "date": pd.Timestamp("2026-03-02"),
            "ticker": "TEST",
            "expiration": [pd.Timestamp(r[0]) for r in rows],
            "strike": [float(r[1]) for r in rows],
            "right": [r[2] for r in rows],
            "bid": [float(r[3]) for r in rows],
            "ask": [float(r[4]) for r in rows],
            "iv": 0.30,
        }
    )


EARNINGS = pd.Timestamp("2026-03-05 21:05:00")  # AMC on Mar 5 (tz-naive)


# ---------- pick_expiry ----------


def test_pick_expiry_nearest_strictly_after_earnings() -> None:
    chain = _chain(
        [
            ("2026-03-05", 100, "call", 1, 2),  # ON the earnings date -> excluded (expires
            ("2026-03-13", 100, "call", 1, 2),  # before the AMC move is realized)
            ("2026-03-20", 100, "call", 1, 2),
        ]
    )
    assert rc.pick_expiry(chain, EARNINGS) == pd.Timestamp("2026-03-13")


def test_pick_expiry_none_when_all_expire_before() -> None:
    chain = _chain([("2026-03-04", 100, "call", 1, 2)])
    assert rc.pick_expiry(chain, EARNINGS) is None


def test_pick_expiry_none_on_empty_chain() -> None:
    assert rc.pick_expiry(_chain([]), EARNINGS) is None


# ---------- pick_atm_strike ----------


def test_pick_atm_strike_requires_both_legs_quoted() -> None:
    chain = _chain(
        [
            ("2026-03-13", 100, "call", 1.0, 1.2),  # call only -> not eligible
            ("2026-03-13", 105, "call", 1.0, 1.2),
            ("2026-03-13", 105, "put", 1.0, 1.2),  # both legs -> eligible
        ]
    )
    assert rc.pick_atm_strike(chain, pd.Timestamp("2026-03-13"), spot=100.0) == 105.0


def test_pick_atm_strike_zero_bid_disqualifies() -> None:
    chain = _chain(
        [
            ("2026-03-13", 100, "call", 0.0, 1.2),  # zero entry bid -> unquoted
            ("2026-03-13", 100, "put", 1.0, 1.2),
        ]
    )
    assert rc.pick_atm_strike(chain, pd.Timestamp("2026-03-13"), spot=100.0) is None


def test_pick_atm_strike_nearest_to_spot() -> None:
    rows = []
    for strike in (95, 100, 105):
        rows.append(("2026-03-13", strike, "call", 1.0, 1.2))
        rows.append(("2026-03-13", strike, "put", 1.0, 1.2))
    assert rc.pick_atm_strike(_chain(rows), pd.Timestamp("2026-03-13"), spot=101.0) == 100.0


# ---------- run_event ----------


def _bars() -> pd.Series:
    idx = pd.bdate_range("2026-02-23", "2026-03-10")
    return pd.Series(100.0, index=idx, name="close")


def _entry_chain(call_bid=4.8, call_ask=5.0, put_bid=3.8, put_ask=4.0) -> pd.DataFrame:
    return _chain(
        [
            ("2026-03-13", 100, "call", call_bid, call_ask),
            ("2026-03-13", 100, "put", put_bid, put_ask),
        ]
    )


def _exit_chain(call_bid=3.0, put_bid=2.0) -> pd.DataFrame:
    return _chain(
        [
            ("2026-03-13", 100, "call", call_bid, call_bid + 0.2),
            ("2026-03-13", 100, "put", put_bid, put_bid + 0.2),
        ]
    )


def _loader(entry: pd.DataFrame, exit_: pd.DataFrame):
    """Earnings bar for the Mar-5 AMC event is Mar 6 (first bar >= 21:05), so
    entry (3 bars before) is 2026-03-03 and exit (1 bar after) is 2026-03-09."""

    def load_chain(ticker: str, when, **kwargs) -> pd.DataFrame:
        return entry if pd.Timestamp(when) <= pd.Timestamp("2026-03-03") else exit_

    return load_chain


def _run(entry, exit_, **kwargs):
    defaults = dict(
        ticker="TEST",
        close=_bars(),
        earnings_datetime=EARNINGS,
        prior_moves=[0.10] * 8,  # expected_move = 0.10
        load_chain=_loader(entry, exit_),
    )
    defaults.update(kwargs)
    return rc.run_event(**defaults)


def test_run_event_quote_to_quote_pnl_arithmetic() -> None:
    rec = _run(_entry_chain(), _exit_chain())

    assert "skip_reason" not in rec
    assert rec["entry_cost"] == pytest.approx(9.0)  # call_ask 5.0 + put_ask 4.0
    assert rec["exit_value"] == pytest.approx(5.0)  # call_bid 3.0 + put_bid 2.0
    # fees = (9 + 5) * 100 * 1bp = 0.14 ; pnl = -400 - 0.14
    assert rec["pnl"] == pytest.approx(-400.14)
    # implied move from MIDs: ((4.9) + (3.9)) / 100 = 0.088
    assert rec["implied_move"] == pytest.approx(0.088)
    # gate: expected 0.10 > 0.088 * 1.2 = 0.1056 ? NO -> gate must not fire
    assert rec["gate_fired"] is False


def test_run_event_gate_fires_on_real_edge() -> None:
    rec = _run(_entry_chain(), _exit_chain(), prior_moves=[0.15] * 8)
    # expected 0.15 > 0.088 * 1.2 = 0.1056 -> fires
    assert rec["gate_fired"] is True


def test_run_event_nan_expected_trades_unfiltered_but_gate_off() -> None:
    rec = _run(_entry_chain(), _exit_chain(), prior_moves=[])
    assert "skip_reason" not in rec  # unfiltered branch keeps forecast-less events
    assert math.isnan(rec["expected_move"])
    assert rec["gate_fired"] is False


def test_run_event_zero_exit_bid_is_total_loss_not_skip() -> None:
    rec = _run(_entry_chain(), _exit_chain(call_bid=0.0, put_bid=0.0))
    assert "skip_reason" not in rec
    assert rec["exit_value"] == 0.0
    # fees = (9 + 0) * 100 * 1bp = 0.09 ; pnl = -900 - 0.09
    assert rec["pnl"] == pytest.approx(-900.09)


def test_run_event_skip_no_entry_chain() -> None:
    rec = _run(_chain([]), _exit_chain())
    assert rec["skip_reason"] == rc.SKIP_NO_ENTRY_CHAIN


def test_run_event_skip_no_post_earnings_expiry() -> None:
    entry = _chain([("2026-03-04", 100, "call", 1, 2), ("2026-03-04", 100, "put", 1, 2)])
    rec = _run(entry, _exit_chain())
    assert rec["skip_reason"] == rc.SKIP_NO_EXPIRY


def test_run_event_skip_no_quoted_atm() -> None:
    entry = _chain([("2026-03-13", 100, "call", 1, 2)])  # missing put leg
    rec = _run(entry, _exit_chain())
    assert rec["skip_reason"] == rc.SKIP_NO_QUOTED_ATM


def test_run_event_skip_spread_over_cap() -> None:
    # bid/ask: call 1.0/3.0, put 1.0/3.0 -> mid premium 4.0, spread 4.0 -> frac 1.0 > 0.20
    entry = _chain(
        [("2026-03-13", 100, "call", 1.0, 3.0), ("2026-03-13", 100, "put", 1.0, 3.0)]
    )
    rec = _run(entry, _exit_chain())
    assert rec["skip_reason"] == rc.SKIP_SPREAD


def test_run_event_skip_no_exit_chain() -> None:
    rec = _run(_entry_chain(), _chain([]))
    assert rec["skip_reason"] == rc.SKIP_NO_EXIT_CHAIN


def test_run_event_window_out_of_range_raises() -> None:
    close = _bars().iloc[-4:]  # not enough bars before earnings for entry_lead=3
    with pytest.raises(ValueError, match="window"):
        _run(_entry_chain(), _exit_chain(), close=close)


# ---------- past_abs_moves consistency with signal.expected_move ----------


def test_past_abs_moves_matches_signal_expected_move() -> None:
    idx = pd.bdate_range("2025-01-01", periods=120)
    close = pd.Series(
        [100 + (i % 7) * 1.5 for i in range(120)], index=idx, name="close", dtype=float
    )
    earnings = pd.Series(pd.to_datetime(["2025-02-03 21:05", "2025-04-01 21:05"]))

    moves = rc.past_abs_moves(close, earnings)
    via_signal = sig.expected_move(close, earnings, lookback=8)

    assert len(moves) == 2
    assert pd.Series(moves[-8:]).mean() == pytest.approx(via_signal)


# ---------- gate_pnls sweep helper ----------


def test_gate_pnls_recomputes_gate_from_stored_rows() -> None:
    events = [
        {"implied_move": 0.05, "prior_moves": [0.10] * 8, "pnl": 100.0},  # 0.10 > 0.06 -> fires
        {"implied_move": 0.20, "prior_moves": [0.10] * 8, "pnl": -50.0},  # 0.10 < 0.24 -> no
    ]
    assert rc.gate_pnls(events, k=1.2, lookback=8) == [100.0]
    # tighter k excludes everything
    assert rc.gate_pnls(events, k=2.5, lookback=8) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_earnings_real_chain.py -v`
Expected: FAIL at collection with `FileNotFoundError` (real_chain.py does not exist)

- [ ] **Step 3: Write the implementation**

Create `models/options/03-earnings-straddle-spy/real_chain.py`:

```python
"""Quote-to-quote real-chain event mechanics for the earnings straddle (Phase 2).

P&L contains NO pricing model: entry buys the ATM call+put at the real ask,
exit sells both at the real bid. A zero (or null) exit bid is a valid
total-loss outcome — only absent contract rows at exit count as missing data.
The k-gate consumes the real implied move ``(call_mid + put_mid) / spot`` via
the unchanged ``signal.py`` functions, so the gate is name-specific for the
first time (Phase 1's synthetic implied move was ~0.075 for every ticker).

Skip reasons are first-class and mutually exclusive; the runner counts and
reports every one (no silent truncation). Deviation from the design spec,
deliberate: ``signal.tradeable_event`` is NOT used because it returns False
when ``expected`` is NaN (no prior earnings history), which would silently
drop forecast-less events from the *unfiltered* baseline; the spread cap is
checked directly and ``signal.passes_filter`` already maps NaN -> gate off.

Timing mirrors ``strategy.py``: the earnings bar is the first bar at/after the
earnings datetime; entry is ``entry_lead`` bars before it, exit is
``exit_offset`` bars after it. Expiry selection is *strictly after* the
earnings datetime: a same-day expiry settles at that day's close, before an
AMC announcement move is realized, so it cannot capture the event.

Fees mirror Phase 1's ``fee_bps=1.0``: each crossing pays fee_bps on its
notional, so ``fees = (entry_cost + exit_value) * 100 * fee_bps / 1e4``.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from tradinglib.options.instruments import CONTRACT_MULTIPLIER

_HERE = Path(__file__).resolve().parent

SKIP_NO_ENTRY_CHAIN = "no_entry_chain"
SKIP_NO_EXPIRY = "no_post_earnings_expiry"
SKIP_NO_QUOTED_ATM = "no_quoted_atm"
SKIP_SPREAD = "spread_over_cap"
SKIP_NO_EXIT_CHAIN = "no_exit_chain"

SKIP_REASONS = [
    SKIP_NO_ENTRY_CHAIN,
    SKIP_NO_EXPIRY,
    SKIP_NO_QUOTED_ATM,
    SKIP_SPREAD,
    SKIP_NO_EXIT_CHAIN,
]

_MAX_LOOKBACK = 20  # prior_moves stored per event are trimmed to this for sweeps


def _load_signal():
    spec = importlib.util.spec_from_file_location("es_signal", _HERE / "signal.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SIG = _load_signal()


def pick_expiry(chain: pd.DataFrame, earnings_datetime: pd.Timestamp) -> pd.Timestamp | None:
    """Nearest available expiration STRICTLY after the earnings datetime."""
    if chain.empty:
        return None
    after = sorted(e for e in set(chain["expiration"]) if e > earnings_datetime)
    return after[0] if after else None


def pick_atm_strike(chain: pd.DataFrame, expiry: pd.Timestamp, spot: float) -> float | None:
    """Nearest strike to spot with BOTH legs quoted (bid > 0 and ask > 0 on call AND put)."""
    sub = chain[chain["expiration"] == expiry]
    quoted = sub[(sub["bid"] > 0) & (sub["ask"] > 0)]
    if quoted.empty:
        return None
    rights_by_strike = quoted.groupby("strike")["right"].agg(set)
    both = [s for s, rights in rights_by_strike.items() if rights >= {"call", "put"}]
    if not both:
        return None
    return float(min(both, key=lambda s: abs(s - spot)))


def straddle_quotes(
    chain: pd.DataFrame, expiry: pd.Timestamp, strike: float
) -> dict[str, float] | None:
    """{call_bid, call_ask, put_bid, put_ask} for one contract pair, or None if a leg is absent."""
    sub = chain[(chain["expiration"] == expiry) & (chain["strike"] == strike)]
    out: dict[str, float] = {}
    for right in ("call", "put"):
        leg = sub[sub["right"] == right]
        if leg.empty:
            return None
        out[f"{right}_bid"] = float(leg["bid"].iloc[0])
        out[f"{right}_ask"] = float(leg["ask"].iloc[0])
    return out


def past_abs_moves(close: pd.Series, earnings_datetimes: pd.Series) -> list[float]:
    """Chronological abs earnings-day returns — the same definition as
    ``signal.expected_move`` (close at the last bar <= the event to the next
    bar), returned as a list so lookback sweeps can re-slice it."""
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
        prev = float(closes.iloc[pos])
        if prev <= 0.0:
            continue
        ret = float(closes.iloc[pos + 1]) / prev - 1.0
        if not math.isfinite(ret):
            continue
        moves.append(abs(ret))
    return moves


def _bid_or_zero(x: float) -> float:
    """A null/negative bid means no resting bid — worth zero to close into."""
    return 0.0 if (x is None or math.isnan(x) or x < 0) else x


def run_event(
    *,
    ticker: str,
    close: pd.Series,
    earnings_datetime: pd.Timestamp,
    prior_moves: list[float],
    load_chain: Callable[..., pd.DataFrame],
    entry_lead: int = 3,
    exit_offset: int = 1,
    k: float = 1.2,
    lookback: int = 8,
    max_spread_frac: float = 0.20,
    fee_bps: float = 1.0,
) -> dict[str, Any]:
    """One earnings event, quote-to-quote.

    Returns ``{"skip_reason": <reason>}`` when the chain cannot trade, else a
    full trade record (the gate is evaluated but pnl is always present, so the
    unfiltered branch can use every tradeable event). Raises ValueError when
    the event window does not fit the bar series (the runner's concern).
    ``close`` must be tz-naive; ``prior_moves`` are abs moves of PRIOR events
    only (leakage discipline is the caller's contract, as in Phase 1).
    """
    bars = close.index
    after = bars >= earnings_datetime
    if not after.any():
        raise ValueError(f"event window out of range: earnings {earnings_datetime} beyond bars")
    e_idx = int(after.argmax())
    entry_idx, exit_idx = e_idx - entry_lead, e_idx + exit_offset
    if entry_idx < 0 or exit_idx >= len(bars):
        raise ValueError(
            f"event window out of range: need {entry_lead} bars before and "
            f"{exit_offset} after the earnings bar"
        )
    entry_date, exit_date = bars[entry_idx], bars[exit_idx]
    spot = float(close.iloc[entry_idx])

    entry_chain = load_chain(ticker, entry_date)
    if entry_chain.empty:
        return {"skip_reason": SKIP_NO_ENTRY_CHAIN}
    expiry = pick_expiry(entry_chain, earnings_datetime)
    if expiry is None:
        return {"skip_reason": SKIP_NO_EXPIRY}
    strike = pick_atm_strike(entry_chain, expiry, spot)
    if strike is None:
        return {"skip_reason": SKIP_NO_QUOTED_ATM}
    quotes = straddle_quotes(entry_chain, expiry, strike)
    if quotes is None:
        return {"skip_reason": SKIP_NO_QUOTED_ATM}

    entry_cost = quotes["call_ask"] + quotes["put_ask"]
    entry_bid = quotes["call_bid"] + quotes["put_bid"]
    mid = (entry_cost + entry_bid) / 2.0
    spread_frac = (entry_cost - entry_bid) / mid if mid > 0 else math.inf
    if spread_frac > max_spread_frac:
        return {"skip_reason": SKIP_SPREAD}

    implied = _SIG.implied_move(mid, spot)
    trimmed = [abs(m) for m in prior_moves][-_MAX_LOOKBACK:]
    window = trimmed[-lookback:]
    expected = float(pd.Series(window).mean()) if window else float("nan")

    exit_chain = load_chain(ticker, exit_date)
    exit_quotes = None if exit_chain.empty else straddle_quotes(exit_chain, expiry, strike)
    if exit_quotes is None:
        return {"skip_reason": SKIP_NO_EXIT_CHAIN}
    exit_value = _bid_or_zero(exit_quotes["call_bid"]) + _bid_or_zero(exit_quotes["put_bid"])

    fees = (entry_cost + exit_value) * CONTRACT_MULTIPLIER * (fee_bps / 1e4)
    pnl = (exit_value - entry_cost) * CONTRACT_MULTIPLIER - fees

    return {
        "ticker": ticker,
        "date": str(pd.Timestamp(earnings_datetime).date()),
        "entry_date": str(pd.Timestamp(entry_date).date()),
        "exit_date": str(pd.Timestamp(exit_date).date()),
        "expiry": str(pd.Timestamp(expiry).date()),
        "strike": float(strike),
        "spot": spot,
        "entry_cost": entry_cost,
        "exit_value": exit_value,
        "spread_frac": spread_frac,
        "implied_move": implied,
        "expected_move": expected,
        "prior_moves": trimmed,
        "gate_fired": bool(_SIG.passes_filter(expected, implied, k)),
        "pnl": pnl,
    }


def gate_pnls(events: list[dict], *, k: float, lookback: int) -> list[float]:
    """Recompute gate membership from stored rows (pnl is k/lookback-independent)."""
    fired: list[float] = []
    for ev in events:
        window = ev["prior_moves"][-lookback:]
        expected = float(pd.Series(window).mean()) if window else float("nan")
        if _SIG.passes_filter(expected, ev["implied_move"], k):
            fired.append(float(ev["pnl"]))
    return fired
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_earnings_real_chain.py -v`
Expected: 18 passed

If `test_run_event_quote_to_quote_pnl_arithmetic` fails on `implied_move`,
check the mid arithmetic: call mid 4.9 + put mid 3.9 = 8.8, spot 100 → 0.088.

- [ ] **Step 5: Run the existing model tests to confirm nothing regressed**

Run: `uv run pytest tests/test_earnings_signal.py tests/test_earnings_strategy.py tests/test_earnings_loader.py -v`
Expected: all pass (real_chain.py touches nothing existing)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check models/options/03-earnings-straddle-spy/real_chain.py tests/test_earnings_real_chain.py
git add models/options/03-earnings-straddle-spy/real_chain.py tests/test_earnings_real_chain.py
git commit -m "feat: quote-to-quote real-chain event mechanics for earnings straddle"
```

---

### Task 4: Universe runner script

**Files:**
- Create: `scripts/earnings_straddle_real_chain_backtest.py`

The runner is glue over already-tested parts (loaders, `run_event`, `gate_pnls`,
and the existing `bootstrap_t_test` / `benjamini_hochberg_fdr` / `trade_metrics`),
mirroring `scripts/earnings_straddle_thorough_backtest.py`. No new unit tests —
the verification is Task 5's live run.

- [ ] **Step 1: Write the runner**

Create `scripts/earnings_straddle_real_chain_backtest.py`:

```python
"""Real-chain (Phase 2) backtest of model 03 — earnings event-vol straddle.

Quote-to-quote on DoltHub historical chains: entry buys the ATM straddle at
the real ask, exit sells at the real bid; the k-gate consumes the REAL implied
move (call_mid + put_mid)/spot, so the selection thesis is testable for the
first time (Phase 1's synthetic implied move was ~0.075 for every name).

Universe/window mirror the Phase-1 thorough backtest for comparability.
Skip reasons are counted and reported — no silent truncation. First live run
fetches ~2 chains per event from the DoltHub API (cached to parquet forever
after; politeness sleep 0.5 s per live call, so expect ~5-10 minutes cold).

Writes models/options/03-earnings-straddle-spy/results/real_chain_backtest.json
and prints a compact summary.

    uv run python scripts/earnings_straddle_real_chain_backtest.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglib.backtest.metrics import benjamini_hochberg_fdr, bootstrap_t_test, trade_metrics
from tradinglib.loaders.equities.yfinance import load_daily
from tradinglib.loaders.events.earnings import get_earnings_dates
from tradinglib.loaders.options.dolthub import load_chain

_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "options" / "03-earnings-straddle-spy"
)

UNIVERSE = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "NFLX", "AMD"]
START = "2020-01-01"
END = "2026-06-05"
ENTRY_LEAD = 3
EXIT_OFFSET = 1
K = 1.2
LOOKBACK = 8
MAX_SPREAD_FRAC = 0.20
FEE_BPS = 1.0
K_GRID = [1.05, 1.2, 1.5, 2.0]
LOOKBACK_GRID = [4, 8, 12]


def _load_real_chain_module():
    spec = importlib.util.spec_from_file_location("es_real_chain", _MODEL_DIR / "real_chain.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load_real_chain_module()


def _pool(pnls: list[float]) -> dict:
    s = pd.Series(pnls, dtype=float)
    tm = trade_metrics(s)
    t_stat, lo, hi, p = bootstrap_t_test(s, n_boot=2000, seed=0)
    return {
        "n": len(pnls),
        "total_pnl": float(s.sum()) if len(pnls) else 0.0,
        "trade_metrics": tm,
        "bootstrap": {"t_stat": t_stat, "ci_lo": lo, "ci_hi": hi, "p_value": p},
    }


def main() -> None:
    events: list[dict] = []
    skips: dict[str, int] = dict.fromkeys(rc.SKIP_REASONS, 0)
    n_window_skips = 0

    for ticker in UNIVERSE:
        bars = load_daily(ticker, START, END)
        close = bars["close"]
        close = pd.Series(close.to_numpy(), index=close.index.tz_localize(None), name="close")
        ev = get_earnings_dates([ticker], start=START, end=END)["earnings_datetime"]
        ev = pd.to_datetime(ev, utc=True).sort_values()

        for traded_aware in ev:
            traded = traded_aware.tz_localize(None)
            prior = ev[ev < traded_aware]
            prior_moves = rc.past_abs_moves(close, prior)
            try:
                rec = rc.run_event(
                    ticker=ticker,
                    close=close,
                    earnings_datetime=traded,
                    prior_moves=prior_moves,
                    load_chain=load_chain,
                    entry_lead=ENTRY_LEAD,
                    exit_offset=EXIT_OFFSET,
                    k=K,
                    lookback=LOOKBACK,
                    max_spread_frac=MAX_SPREAD_FRAC,
                    fee_bps=FEE_BPS,
                )
            except ValueError:
                n_window_skips += 1
                continue
            if "skip_reason" in rec:
                skips[rec["skip_reason"]] += 1
                continue
            events.append(rec)
        print(f"{ticker}: {sum(1 for e in events if e['ticker'] == ticker)} events traded")

    filtered = [e["pnl"] for e in events if e["gate_fired"]]
    unfiltered = [e["pnl"] for e in events]

    per_ticker: dict[str, dict] = {}
    for ticker in UNIVERSE:
        t_events = [e for e in events if e["ticker"] == ticker]
        per_ticker[ticker] = {
            "n_events": len(t_events),
            "n_gate_fired": sum(e["gate_fired"] for e in t_events),
            "mean_implied_move": float(np.nanmean([e["implied_move"] for e in t_events]))
            if t_events
            else float("nan"),
            "filtered": _pool([e["pnl"] for e in t_events if e["gate_fired"]]),
            "unfiltered": _pool([e["pnl"] for e in t_events]),
        }

    fdr_tickers = [t for t in UNIVERSE if per_ticker[t]["filtered"]["n"] >= 2]
    pvals = [per_ticker[t]["filtered"]["bootstrap"]["p_value"] for t in fdr_tickers]
    rejected, threshold = benjamini_hochberg_fdr(pvals, alpha=0.05)

    sweep = []
    for k in K_GRID:
        for lb in LOOKBACK_GRID:
            pnls = rc.gate_pnls(events, k=k, lookback=lb)
            pool = _pool(pnls)
            sweep.append(
                {
                    "k": k,
                    "lookback": lb,
                    "n_fired": pool["n"],
                    "expectancy": pool["trade_metrics"]["expectancy"],
                    "win_rate": pool["trade_metrics"]["win_rate"],
                    "profit_factor": pool["trade_metrics"]["profit_factor"],
                    "total_pnl": pool["total_pnl"],
                    "p_value": pool["bootstrap"]["p_value"],
                }
            )

    out = {
        "phase": "2-real-chain-dolthub",
        "source": "dolthub post-no-preference/options (EOD quotes)",
        "universe": UNIVERSE,
        "window": [START, END],
        "params": {
            "k": K,
            "lookback": LOOKBACK,
            "entry_lead": ENTRY_LEAD,
            "exit_offset": EXIT_OFFSET,
            "max_spread_frac": MAX_SPREAD_FRAC,
            "fee_bps": FEE_BPS,
        },
        "n_events_traded": len(events),
        "n_gate_fired": len(filtered),
        "skips": {**skips, "window_out_of_range": n_window_skips},
        "pooled_filtered": _pool(filtered),
        "pooled_unfiltered": _pool(unfiltered),
        "fdr": {
            "tickers": fdr_tickers,
            "p_values": pvals,
            "rejected": rejected,
            "threshold": threshold,
        },
        "per_ticker": per_ticker,
        "sensitivity_k_lookback": sweep,
        "events": events,
    }

    out_path = _MODEL_DIR / "results" / "real_chain_backtest.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))

    def line(label: str, pool: dict) -> None:
        tm, b = pool["trade_metrics"], pool["bootstrap"]
        print(
            f"{label:>22s}: n={pool['n']:>3d}  exp=${tm['expectancy']:>9.2f}  "
            f"win={tm['win_rate']:.2%}  PF={tm['profit_factor']:.2f}  "
            f"total=${pool['total_pnl']:>11.2f}  p={b['p_value']:.3f}"
        )

    print(f"\n=== REAL-CHAIN BACKTEST  events={len(events)}  gate_fired={len(filtered)} ===")
    print(f"skips: {out['skips']}")
    line("POOLED filtered", out["pooled_filtered"])
    line("POOLED unfiltered", out["pooled_unfiltered"])
    print("\n--- per ticker (mean REAL implied move — must differ across names) ---")
    for t in UNIVERSE:
        pt = per_ticker[t]
        print(
            f"{t:>6s}: events={pt['n_events']:>2d} fired={pt['n_gate_fired']:>2d} "
            f"mean_implied={pt['mean_implied_move']:.4f}"
        )
    print(f"\nFDR: tickers={fdr_tickers} rejected={rejected} threshold={threshold}")
    print("\n--- sensitivity: k x lookback ---")
    for r in sweep:
        print(
            f"  k={r['k']:.2f} lb={r['lookback']:>2d}: fired={r['n_fired']:>3d} "
            f"exp=${r['expectancy']:>8.2f} win={r['win_rate']:.2%} "
            f"PF={r['profit_factor']:.2f} total=${r['total_pnl']:>10.2f} p={r['p_value']:.3f}"
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint and sanity-import**

Run: `uv run ruff check scripts/earnings_straddle_real_chain_backtest.py`
Expected: clean.

Run: `uv run python -c "import importlib.util, pathlib; p = pathlib.Path('scripts/earnings_straddle_real_chain_backtest.py'); s = importlib.util.spec_from_file_location('m', p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print('imports ok')"`
Expected: `imports ok` (module-level code loads `real_chain` and all tradinglib imports without running `main()`).

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest`
Expected: all tests pass (the runner adds no test surface, but the import graph must stay healthy).

- [ ] **Step 4: Commit**

```bash
git add scripts/earnings_straddle_real_chain_backtest.py
git commit -m "feat: real-chain universe runner for earnings straddle (quote-to-quote, FDR, k x lookback sweep)"
```

---

### Task 5: First live run (network required)

**Files:**
- Generated: `models/options/03-earnings-straddle-spy/results/real_chain_backtest.json`
- Generated: `data/processed/options/dolthub/<ticker>/<date>.parquet` (~430 files)
- Generated: `data/processed/options/yf_snapshots/<ticker>/<today>.parquet` (9 files)

This task hits live APIs (DoltHub, yfinance). It is the verification of Tasks 1–4.

- [ ] **Step 1: Run the snapshot collector**

Run: `uv run python scripts/collect_chain_snapshots.py`
Expected: 9 lines, one per watchlist ticker, each `TICKER: <n> rows` with n > 0.

Run it AGAIN immediately: every line must read `already snapshotted today` (idempotency, live).

- [ ] **Step 2: Run the real-chain backtest (cold cache: expect 5–10 minutes)**

Run: `uv run python scripts/earnings_straddle_real_chain_backtest.py`
Expected: per-ticker progress lines, then the summary block, then `wrote ...real_chain_backtest.json`.

- [ ] **Step 3: Verify the run is sound (not the result — the mechanics)**

Check each of these in the console output / JSON:

1. **Implied moves are name-specific.** The per-ticker `mean_implied` values must
   visibly differ across names (e.g. MSFT well below TSLA). If they cluster
   around one value like Phase 1's ~0.075, the gate wiring is broken — stop and debug.
2. **Skip accounting adds up.** `n_events_traded + sum(skips.values())` equals the
   total usable earnings events. A large `no_entry_chain` count (> ~25% of events)
   suggests DoltHub coverage gaps — note the number; do not hide it.
3. **Quotes sanity.** Spot-check 2–3 `events` entries in the JSON against the DoltHub
   web UI (https://www.dolthub.com/repositories/post-no-preference/options) — same
   date/strike/expiry should show the same bid/ask.
4. **Cache works.** Re-run the script; the second run must complete in well under a
   minute (all chains served from parquet).

- [ ] **Step 4: Commit results and caches**

The repo commits processed parquet (see `data/processed/events/earnings/` in git).
Commit the chain caches the same way — they are the reproducibility substrate:

```bash
git add data/processed/options models/options/03-earnings-straddle-spy/results/real_chain_backtest.json
git commit -m "data: DoltHub chain caches + first real-chain backtest results"
```

(If the parquet payload turns out unexpectedly large — check with `git status` /
folder size first — commit only `real_chain_backtest.json` and add
`data/processed/options/dolthub/` to `.gitignore` instead, noting the choice in
the commit message.)

---

### Task 6: Documentation and re-verdict

**Files:**
- Modify: `models/options/03-earnings-straddle-spy/model.md`
- Modify: `models/options/03-earnings-straddle-spy/README.md`
- Modify: `docs/data-sources.md`
- Modify: `docs/specs/2026-06-09-earnings-straddle-real-chain-design.md` (one amendment)

The numbers come from Task 5's `real_chain_backtest.json` — fill them in from the
actual output; never estimate them.

- [ ] **Step 1: Amend the spec for the tradeable_event deviation**

In `docs/specs/2026-06-09-earnings-straddle-real-chain-design.md`, in the
Component-3 gate bullet, replace the sentence claiming chain-tradeability goes
via `signal.tradeable_event` with:

> Chain-tradeability is checked directly (spread cap `max_spread_frac=0.20` on
> `(asks − bids) / mid premium`); `signal.tradeable_event` is deliberately NOT
> used because it returns False on NaN `expected` (no prior history), which
> would drop forecast-less events from the *unfiltered* baseline. The gate
> itself goes through `signal.passes_filter`, which maps NaN → no-fire.

- [ ] **Step 2: Add the Phase-2 section to `model.md`**

Insert after the Phase-1 "Thorough backtest" section a new section:

```markdown
## Phase 2 — real chain (DoltHub), quote-to-quote

Reproduce: `uv run python scripts/earnings_straddle_real_chain_backtest.py`
(writes `results/real_chain_backtest.json`; chains cached under
`data/processed/options/dolthub/`).

Pricing contains **no model**: entry buys the ATM straddle at the real ask,
exit sells at the real bid (EOD quotes); the implied move is the real
`(call_mid + put_mid)/spot` at entry — name-specific for the first time, so
the k-gate is finally a mispricing test rather than a realized-vol screen.
Expiry is the nearest *available* listed expiration strictly after earnings
(the dataset carries 3–4 expirations per day, ~2–7 weeks out). Fees 1 bp per
crossing; spread cap 20% of mid premium; skips are counted, not hidden.

### Headline (<n> events, k=1.2, lookback=8)

| Branch | n | Expectancy | Median | Win | PF | Total | Bootstrap p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unfiltered | <from JSON> | ... | ... | ... | ... | ... | ... |
| Filtered (k-gate) | <from JSON> | ... | ... | ... | ... | ... | ... |

(The Median column is not in `trade_metrics` — compute it from the per-event
`pnl` values in the JSON's `events` array: all events for unfiltered, the
`gate_fired` subset for filtered.)

Skip accounting: <skips dict from JSON>. FDR: <fdr from JSON>.

### Findings

<Write 3-6 findings from the actual numbers: does the unfiltered VRP bleed
persist on real quotes? Is the filtered branch significant? Which names drive
it? How does the k x lookback sweep behave now that there is no pre_iv to
sweep? Be as blunt as the Phase-1 write-up.>
```

Update the front-matter `status:` and the **Viability** section ONLY as the real
numbers warrant — a negative result keeps `status: negative-result` and gets the
same honest treatment as Phase 1.

- [ ] **Step 3: Update `README.md`**

Mirror the model.md changes in compressed form: replace the "Phase 1 is synthetic —
NOT yet tradeable" framing with a short "Phase 2 — real chain" section (the
synthetic phase remains documented above it), the headline table, and a pointer to
`real_chain_backtest.json`. Update the "Deferred" list: listed-expiry snap is now
DONE (real expirations); still deferred: BMO/AMC-aware exit timing, options-aware
walk-forward / OOS seasons, short-premium mirror, consuming yf snapshots.

- [ ] **Step 4: Update `docs/data-sources.md`**

Add two entries following the existing format:

1. **DoltHub options (historical chains)** — `tradinglib/loaders/options/dolthub.py`;
   post-no-preference/options via the free SQL API; EOD bid/ask/IV; PK-prefix query
   discipline (`date` AND `act_symbol` or the query times out); only 3–4 expirations
   per (date, symbol); cache layout; coverage ~2019 → present (verified live through
   2026-06-08).
2. **yfinance chain snapshots (forward)** — `tradinglib/loaders/options/yf_chain.py` +
   `scripts/collect_chain_snapshots.py`; canonical schema + `spot`; idempotent per
   day; intended daily cadence (Windows Task Scheduler example:
   `schtasks /create /tn chain-snapshots /tr "uv run python scripts/collect_chain_snapshots.py" /sc daily /st 16:30`);
   point-in-time OOS substrate for the earnings straddle.

- [ ] **Step 5: Run the full suite one last time and commit**

```bash
uv run pytest
uv run ruff check .
git add models/options/03-earnings-straddle-spy/model.md models/options/03-earnings-straddle-spy/README.md docs/data-sources.md docs/specs/2026-06-09-earnings-straddle-real-chain-design.md
git commit -m "docs: Phase-2 real-chain results and re-verdict for earnings straddle"
```

---

## Self-Review Notes (already applied)

- **Spec coverage:** Component 1 → Task 1; Component 2 → Task 2; Component 3 → Tasks 3–5; Component 4 → Task 6; success criteria map to Task 4 step 3, Task 5 steps 1–3, Task 6.
- **Type consistency:** `load_chain(ticker, when, *, refresh=False)` is the loader signature; `run_event` takes it as the injectable `load_chain` callable and calls it `load_chain(ticker, entry_date)` — positionally compatible. `gate_pnls(events, *, k, lookback)` matches the runner's call. `SKIP_*` constants defined in Task 3 are the ones the runner counts in Task 4.
- **Known judgment calls encoded above:** NaN exit bid → 0.0 (no resting bid); same-day expiry excluded (strictly-after rule, AMC rationale); forecast-less events stay in the unfiltered branch (spec amendment in Task 6 step 1).
