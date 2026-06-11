# Planner Sizing Settings + NaN-Bar Loader Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the /planner chat from asking for account size / risk% (page-level settings strip) and fix the yfinance NaN-price trailing row that breaks `propose_trade_levels` with "degenerate ATR".

**Architecture:** Client-owned settings (localStorage on /planner) ride each chat POST as a `settings` object; the server validates and folds one settings line into the opening message (same mechanism as the on-screen backtest `context`); the system prompt then confirms scenario-only. The NaN-bar fix is a single `dropna` on `load_daily`'s read path so it protects every consumer and heals already-poisoned parquet caches.

**Tech Stack:** Python 3.12, FastAPI + Jinja2 templates (vanilla JS, no framework), pandas, pytest, uv.

**Spec:** `docs/specs/2026-06-11-planner-sizing-settings-and-nan-bars-design.md`
**Branch:** `feat/planner-sizing-settings` (stacked on `feat/planner-guided-levels`, PR #66). Retarget the PR to `main` after #66 merges — squash-merge stacks must be retargeted before the base branch is deleted.

**Conventions for this repo:**
- Run everything through `uv run …` (CI does), or the venv at `.venv\Scripts\`.
- The verification gate is: `uv run ruff check .` && `uv run ruff format --check .` && `uv run mypy tradinglib` && `uv run pytest`. `ruff format --check` is separate from `ruff check` — do not skip it.
- Loader tests never hit the network: they stub `yf.download` and redirect `processed_dir` to `tmp_path` (see existing tests in `tests/test_yfinance_loader.py`).
- The chat server is stateless; the browser replays history. Tests capture what the agent saw via `StubProvider.calls`.

---

### Task 1: Drop NaN-price rows in `load_daily`

yfinance currently returns the most recent session with volume populated but NaN open/high/low/close (reproduced on RIVN and SPY, 2026-06-11). That row makes ATR(14) and spot NaN, so the planner's `propose_trade_levels` raises `degenerate ATR (nan)` and the chat falls back to interrogating the user.

**Files:**
- Modify: `tradinglib/loaders/equities/yfinance.py` (the `load_daily` body, after the cache-read/download block, before the start/end filters)
- Test: `tests/test_yfinance_loader.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_yfinance_loader.py`:

```python
@pytest.fixture
def fake_yf_frame_with_nan_tail(fake_yf_frame: pd.DataFrame) -> pd.DataFrame:
    """yfinance glitch: a trailing row with volume populated but NaN prices."""
    df = fake_yf_frame.copy()
    nan_day = pd.Timestamp("2024-01-06")
    for col in ("Open", "High", "Low", "Close"):
        df.loc[nan_day, (col, "SPY")] = float("nan")
    df.loc[nan_day, ("Volume", "SPY")] = 500_000
    return df


def test_load_daily_drops_nan_price_rows(
    fake_yf_frame_with_nan_tail: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.equities import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    with patch.object(loader.yf, "download", return_value=fake_yf_frame_with_nan_tail):
        df = loader.load_daily("SPY")

    assert len(df) == 5  # the NaN-price tail row is gone
    assert df.index.max() == pd.Timestamp("2024-01-05", tz="UTC")
    assert not df[["open", "high", "low", "close"]].isna().any().any()


def test_load_daily_filters_nan_rows_from_poisoned_cache(
    fake_yf_frame_with_nan_tail: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.equities import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    # A cache written before the guard existed still contains the bad row.
    poisoned = loader._canonicalize(fake_yf_frame_with_nan_tail, "SPY")
    out = tmp_path / "yfinance" / "SPY" / "daily.parquet"
    out.parent.mkdir(parents=True)
    poisoned.to_parquet(out)

    df = loader.load_daily("SPY")  # cache hit — no download stub needed
    assert len(df) == 5
    assert not df[["open", "high", "low", "close"]].isna().any().any()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_yfinance_loader.py -v`
Expected: the two new tests FAIL with `assert 6 == 5` (the NaN row survives); the five existing tests still PASS.

- [ ] **Step 3: Implement the filter**

In `tradinglib/loaders/equities/yfinance.py::load_daily`, between the cache-read/download block and the `if start is not None:` filter, insert:

```python
    # yfinance occasionally emits a partial trailing row (volume populated,
    # prices NaN); a bar without prices is unusable downstream, and filtering
    # on the read path also heals caches written before this guard existed.
    df = df.dropna(subset=["open", "high", "low", "close"])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_yfinance_loader.py -v`
Expected: all 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add tradinglib/loaders/equities/yfinance.py tests/test_yfinance_loader.py
git commit -m "fix(loaders): drop NaN-price rows from yfinance dailies"
```

---

### Task 2: `run_chat` folds a sizing-settings line into the opening message

**Files:**
- Modify: `tradinglib/assistant/agent.py:67-93` (the `run_chat` signature, docstring, and opening-message build)
- Test: `tests/test_assistant_agent.py` (append at end — the `_final()` helper used below is already defined mid-file)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_assistant_agent.py`:

```python
def test_settings_folded_into_opening_message():
    provider = StubProvider([_final()])
    _events(
        run_chat(
            "I'm bullish on RIVN",
            provider,
            Budget(),
            settings=(
                "Planner sizing (set on the page): account size $50,000; "
                "risk per trade 2% (0.02)."
            ),
        )
    )
    opening = provider.calls[0][0].text
    assert "I'm bullish on RIVN" in opening
    assert "$50,000" in opening
    assert "do not ask the user for account size or risk" in opening


def test_settings_and_context_both_fold_into_opening():
    provider = StubProvider([_final()])
    _events(
        run_chat(
            "explain",
            provider,
            Budget(),
            context="Backtest: SMA · SPY",
            settings=(
                "Planner sizing (set on the page): account size $100,000; "
                "risk per trade 1% (0.01)."
            ),
        )
    )
    opening = provider.calls[0][0].text
    assert "SMA" in opening and "$100,000" in opening and "explain" in opening
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_assistant_agent.py -v`
Expected: the two new tests FAIL with `TypeError: run_chat() got an unexpected keyword argument 'settings'`; the rest PASS.

- [ ] **Step 3: Implement the parameter**

In `tradinglib/assistant/agent.py`, change the `run_chat` signature:

```python
def run_chat(
    user_message: str,
    provider: LLMProvider,
    budget: Budget,
    context: str | None = None,
    history: Sequence[tuple[str, str]] | None = None,
    settings: str | None = None,
) -> Iterator[dict[str, Any]]:
```

Add to the docstring (after the `history` paragraph):

```
    ``settings`` (optional) is the /planner sizing line (account size, risk per
    trade) the webapp renders from the page's settings strip. When present it
    is appended to the opening message with an instruction to use it for
    sizing and never ask.
```

And after the existing `if context:` block, before `conversation = _seed(...)`:

```python
    if settings:
        opening = (
            f"{opening}\n\n{settings}\n"
            "Use these for sizing; do not ask the user for account size or risk."
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_assistant_agent.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tradinglib/assistant/agent.py tests/test_assistant_agent.py
git commit -m "feat(assistant): fold planner sizing settings into the chat opening"
```

---

### Task 3: `/api/v1/chat` accepts and validates `settings`

**Files:**
- Modify: `webapp/main.py` (new `_planner_settings` helper after `_chat_history` ~line 201; two edits in the `chat` route ~lines 338 and 372)
- Test: `tests/test_webapp_chat.py` (append at end — reuses the `_stub_chat_provider` helper defined mid-file)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_webapp_chat.py`:

```python
# ── planner sizing settings ────────────────────────────────────────────


def test_chat_forwards_planner_settings_to_agent(monkeypatch):
    stub = _stub_chat_provider(monkeypatch)
    client = TestClient(create_app())
    resp = client.post(
        "/api/v1/chat",
        json={
            "message": "I'm bullish on RIVN",
            "settings": {"account_size": 50000, "risk_per_trade_pct": 0.02},
        },
    )
    assert resp.status_code == 200
    opening = stub.calls[0][0].text
    assert "account size $50,000" in opening
    assert "2%" in opening
    assert "do not ask the user for account size or risk" in opening


def test_chat_malformed_settings_ignored_not_400(monkeypatch):
    # Bad settings degrade to the no-settings flow — never block the chat.
    client = TestClient(create_app())
    for bad in (
        "not a dict",
        {"account_size": -5, "risk_per_trade_pct": 0.01},
        {"account_size": 100000, "risk_per_trade_pct": 0.5},
        {"account_size": "lots"},
        None,
    ):
        stub = _stub_chat_provider(monkeypatch)
        resp = client.post("/api/v1/chat", json={"message": "hi", "settings": bad})
        assert resp.status_code == 200, f"rejected {bad!r}"
        assert "Planner sizing" not in stub.calls[0][0].text, f"accepted {bad!r}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_webapp_chat.py -v`
Expected: `test_chat_forwards_planner_settings_to_agent` FAILS (no settings line in the opening). `test_chat_malformed_settings_ignored_not_400` may already PASS (unknown keys are ignored today) — that is fine; it pins the contract.

- [ ] **Step 3: Implement validator + plumbing**

In `webapp/main.py`, after `_chat_history`, add:

```python
def _planner_settings(raw: Any) -> str | None:
    """Render the /planner sizing strip into one opening-message line.

    Returns ``None`` for anything malformed (missing keys, non-numbers,
    out-of-range values) so a bad payload degrades to the no-settings flow
    instead of 400ing the chat.
    """
    if not isinstance(raw, dict):
        return None
    try:
        account = float(raw["account_size"])
        risk = float(raw["risk_per_trade_pct"])
    except (KeyError, TypeError, ValueError):
        return None
    if account <= 0 or not 0 < risk <= 0.2:
        return None
    return (
        f"Planner sizing (set on the page): account size ${account:,.0f}; "
        f"risk per trade {risk * 100:g}% ({risk:g})."
    )
```

In the `chat` route, after `context = _chat_context(payload.get("context"))`:

```python
        settings = _planner_settings(payload.get("settings"))
```

And change the `run_chat` call to:

```python
                for event in run_chat(
                    message,
                    provider,
                    Budget(),
                    context=context,
                    history=history or None,
                    settings=settings,
                ):
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_webapp_chat.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add webapp/main.py tests/test_webapp_chat.py
git commit -m "feat(webapp): accept planner sizing settings on /api/v1/chat"
```

---

### Task 4: Settings strip on /planner + console transport

The strip and the transport ship together (either alone is dead code). The JS has no test runner — the template tests pin that the markup and wiring strings render on the right pages.

**Files:**
- Modify: `webapp/templates/planner.html` (CSS block, markup after the strapline, persistence JS in the bottom script)
- Modify: `webapp/templates/_console.html` (helper next to `currentContext()`, one key in the POST body)
- Test: `tests/test_webapp_routes.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_webapp_routes.py`:

```python
def test_planner_settings_strip_renders():
    client = TestClient(create_app())
    html = client.get("/planner").text
    assert 'id="planner-settings"' in html
    assert 'id="ps-account"' in html and 'id="ps-risk"' in html
    assert "tm-planner-account" in html  # localStorage persistence wired
    assert "plannerSettings" in html  # the composer reads the strip


def test_index_has_no_settings_strip():
    # The strip is /planner-only; the index console sends no settings.
    client = TestClient(create_app())
    assert 'id="planner-settings"' not in client.get("/").text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_webapp_routes.py -v`
Expected: `test_planner_settings_strip_renders` FAILS on `id="planner-settings"`; `test_index_has_no_settings_strip` PASSES (pins the contract); existing tests PASS.

- [ ] **Step 3: Add the strip to `planner.html`**

In the `<style>` block, after the `.strap` rule, add:

```css
    .settings { display: flex; align-items: center; gap: 10px; padding: 8px 18px;
                border-bottom: 1px solid var(--line); }
    .settings .label { font-size: 9px; color: var(--muted); }
    .settings input { width: 110px; padding: 4px 6px; font: inherit; font-size: 12px;
                      background: var(--field); color: var(--ink);
                      border: 1px solid var(--line); border-radius: 0; }
    .settings input:focus { outline: none; border-color: var(--accent); }
    .settings .hint { color: var(--muted); font-size: 10px; }
```

In the body, directly after the `<div class="strap">…</div>` line, add:

```html
  <div class="settings" id="planner-settings">
    <label class="label" for="ps-account">account $</label>
    <input id="ps-account" type="number" min="1" step="1000" value="100000" />
    <label class="label" for="ps-risk">risk/trade %</label>
    <input id="ps-risk" type="number" min="0.1" max="20" step="0.1" value="1.0" />
    <span class="hint">sizes every ticket — the chat won't ask</span>
  </div>
```

In the bottom `<script>` (after the theme IIFE), add:

```javascript
    (function () {
      var acct = document.getElementById("ps-account"),
          risk = document.getElementById("ps-risk");
      try {
        var a = localStorage.getItem("tm-planner-account"),
            r = localStorage.getItem("tm-planner-risk");
        if (a) acct.value = a;
        if (r) risk.value = r;
      } catch (e) {}
      function save() {
        try {
          localStorage.setItem("tm-planner-account", acct.value);
          localStorage.setItem("tm-planner-risk", risk.value);
        } catch (e) {}
      }
      acct.addEventListener("change", save);
      risk.addEventListener("change", save);
    })();
```

- [ ] **Step 4: Add the transport to `_console.html`**

Directly after the `currentContext()` function definition, add:

```javascript
    function plannerSettings() {
      var box = document.getElementById("planner-settings");
      if (!box) return null;
      var acct = parseFloat(document.getElementById("ps-account").value);
      var riskPct = parseFloat(document.getElementById("ps-risk").value);
      if (!isFinite(acct) || acct <= 0 || !isFinite(riskPct) || riskPct <= 0) return null;
      return { account_size: acct, risk_per_trade_pct: riskPct / 100 };
    }
```

In the `fetch("/api/v1/chat", …)` body, add one key after `context`:

```javascript
          body: JSON.stringify({
            message: msg,
            history: history.slice(-20),
            context: currentContext(),
            settings: plannerSettings(),
            provider: providerEl ? providerEl.value : "claude"
          })
```

(`settings: null` on the index page is fine — `_planner_settings(None)` returns `None`.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_webapp_routes.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add webapp/templates/planner.html webapp/templates/_console.html tests/test_webapp_routes.py
git commit -m "feat(webapp): sizing settings strip on /planner"
```

---

### Task 5: Scenario-only confirm in the system prompt

**Files:**
- Modify: `tradinglib/assistant/provider.py:49-54` (planner step 3 of `SYSTEM_PROMPT`)
- Test: `tests/test_assistant_agent.py` (append — guards against the instruction being lost in future prompt edits)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_assistant_agent.py` (add `SYSTEM_PROMPT` to the existing `tradinglib.assistant.provider` import, or import it in the test):

```python
def test_system_prompt_handles_page_sizing_settings():
    from tradinglib.assistant.provider import SYSTEM_PROMPT

    assert "planner sizing settings" in SYSTEM_PROMPT.lower()
    assert "confirm only the scenario" in SYSTEM_PROMPT.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_assistant_agent.py::test_system_prompt_handles_page_sizing_settings -v`
Expected: FAIL on the first assert.

- [ ] **Step 3: Amend the prompt**

In `tradinglib/assistant/provider.py`, replace this part of `SYSTEM_PROMPT` step 3:

```python
    "its note, and name the alternative scenario keys. (3) End with ONE bundled "
    "confirmation: which scenario (or any tweaked number — keep user-supplied "
    "levels when given), account size and risk per trade, defaulting to $100,000 "
    "and 1% (0.01); tell them 'go' accepts the recommendation with the defaults. "
    "Never ask separate questions for entry, stop, target, band, account size, or "
    "risk — bundle what you need into a single short question. (4) On confirmation "
```

with:

```python
    "its note, and name the alternative scenario keys. (3) End with ONE bundled "
    "confirmation. When the conversation includes planner sizing settings (set on "
    "the page), use them for account size and risk, never mention or ask about "
    "sizing, and confirm only the scenario: which scenario (or any tweaked number "
    "— keep user-supplied levels when given); 'go' accepts the recommendation. "
    "Without settings, the same single question also covers account size and risk "
    "per trade, defaulting to $100,000 and 1% (0.01); 'go' accepts the "
    "recommendation with the defaults. Never ask separate questions for entry, "
    "stop, target, band, account size, or risk — bundle what you need into a "
    "single short question. (4) On confirmation "
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_assistant_agent.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tradinglib/assistant/provider.py tests/test_assistant_agent.py
git commit -m "feat(assistant): scenario-only confirm when page sizing settings present"
```

---

### Task 6: Full gate + live verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full verification gate**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy tradinglib
uv run pytest
```

Expected: all four green. If `ruff format --check` flags the new code, run `uv run ruff format .` on the touched files and re-run the gate.

- [ ] **Step 2: Verify the ATR fix against the live feed (network)**

Run: `.venv\Scripts\python.exe data\tmp\repro_atr.py`
Expected: for RIVN and SPY, `atr14` is a finite positive float (no longer `nan`), `NaN counts` are all 0, and the last bar shown is the most recent *complete* session. (This script was used to reproduce the bug pre-fix; it is throwaway in gitignored `data/tmp/` — do not commit it.)

- [ ] **Step 3: Manual smoke of the strip (optional, needs `ANTHROPIC_API_KEY`)**

```bash
uv run uvicorn webapp.main:app --port 8000
```

On `http://localhost:8000/planner`: set account to 50000 / risk to 2, send "I'm bullish on RIVN". Expected: a levels card renders (chart + scenarios + events), the reply confirms **scenario only** (no account/risk question), and after "go" the ticket sizes off $50,000 / 2%. Reload the page — the strip remembers 50000 / 2.

- [ ] **Step 4: Push and open the stacked PR**

```bash
git push -u origin feat/planner-sizing-settings
gh pr create --base feat/planner-guided-levels --title "feat(planner): page-level sizing settings + NaN-bar loader fix" --body "Settings strip on /planner (account size / risk%, localStorage) folded into the chat opening so the assistant never asks for sizing; load_daily drops yfinance's NaN-price trailing rows, fixing propose_trade_levels' degenerate-ATR failure. Spec: docs/specs/2026-06-11-planner-sizing-settings-and-nan-bars-design.md"
```

Note: after #66 merges (squash), retarget this PR to `main` **before** the base branch is deleted.
