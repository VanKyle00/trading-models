"""Streamlit app: browse the trading-models portfolio and run backtests.

Launch locally with::

    uv run streamlit run app/streamlit_app.py

Or open the deployed app on Streamlit Cloud (URL in the repo README).
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from app.adapters import run
from app.model_config import default_ticker, ticker_choices, ticker_mode
from app.models_registry import list_models
from app.presets import presets_for_assets
from app.ui import data_details, data_view, options_view, results_view

st.set_page_config(
    page_title="Trading Models",
    layout="wide",
    page_icon="📈",
    menu_items={
        "Get Help": "https://github.com/VanKyle00/trading-models",
        "About": (
            "**Trading Models** — interactive frontend for the "
            "[trading-models](https://github.com/VanKyle00/trading-models) "
            "portfolio. Pick a model, pick a date range or regime preset, and "
            "see the equity curve, drawdown, and metrics produced by the same "
            "backtest engine each model uses on disk."
        ),
    },
)

DEFAULT_START = date(2022, 1, 1)
DEFAULT_END = date(2024, 12, 31)


# ---------- sidebar -------------------------------------------------------

models = list(list_models())
if not models:
    st.error("No models found. Add one under `models/<family>/<slug>/model.md`.")
    st.stop()

# Seed session state so preset buttons can populate the date inputs.
st.session_state.setdefault("start_date", DEFAULT_START)
st.session_state.setdefault("end_date", DEFAULT_END)

with st.sidebar:
    st.header("Configuration")

    model_labels = {m["name"]: m for m in models}
    selected_name = st.selectbox("Model", list(model_labels.keys()))
    selected = model_labels[selected_name]

    # Resolving the ticker can fail if a model's `tickers:` frontmatter is
    # missing or malformed (e.g. a stale deploy serving outdated model.md, or
    # cached metadata from before the field was added). That must not take down
    # the whole app — the model selectbox above has already rendered, so we
    # surface an actionable error for this model and stop, letting the user
    # pick another model. CI's test_every_model_declares_tickers keeps valid
    # frontmatter a hard requirement, so this only ever fires on a bad deploy.
    mode = ticker_mode(selected)
    try:
        if mode == "free":
            symbol = (
                st.text_input(
                    "Ticker",
                    value=default_ticker(selected),
                    help="Any yfinance symbol — e.g. SPY, QQQ, AAPL, TLT, BTC-USD.",
                )
                .strip()
                .upper()
            )
        elif mode == "choice":
            choices = ticker_choices(selected)
            symbol = st.selectbox("Ticker", choices, index=choices.index(default_ticker(selected)))
        else:  # fixed — locked model
            symbol = default_ticker(selected)
            st.text_input(
                "Ticker",
                value=symbol,
                disabled=True,
                help="This model is locked to its data source and can't switch tickers.",
            )
    except (ValueError, KeyError) as exc:
        st.error(
            f"**{selected['name']}** has no usable ticker configuration "
            "(its `tickers:` frontmatter is missing or malformed). Pick another "
            "model from the dropdown above. If you just deployed an update, "
            "reboot the app (Manage app → Reboot) so it reloads the latest "
            f"`model.md` files.\n\nDetails: `{exc}`"
        )
        st.stop()

    st.divider()

    start = st.date_input("Start date", key="start_date")
    end = st.date_input("End date", key="end_date")

    relevant_presets = presets_for_assets(selected.get("assets"))
    if relevant_presets:
        st.caption("Regime presets")

        def _apply_preset(preset_start: date, preset_end: date) -> None:
            # on_click fires *before* the next rerun, so we can update
            # the date-input session-state keys here. Doing the same
            # write inline after the widget renders triggers
            # StreamlitAPIException: cannot modify widget key after the
            # widget is instantiated.
            st.session_state["start_date"] = preset_start
            st.session_state["end_date"] = preset_end

        for preset in relevant_presets:
            st.button(
                preset.label,
                key=f"preset_{preset.label}",
                on_click=_apply_preset,
                args=(preset.start, preset.end),
                use_container_width=True,
            )

    st.divider()

    run_clicked = st.button("▶ Run backtest", type="primary", use_container_width=True)


# ---------- model card ----------------------------------------------------

st.title("Trading Models")
st.caption(
    "Interactive frontend for the trading-models portfolio. "
    "[Source on GitHub →](https://github.com/VanKyle00/trading-models)"
)
st.subheader(selected["name"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Family", str(selected.get("family", "—")))
c2.metric("Window", str(selected.get("window", "—")))
assets = selected.get("assets") or []
c3.metric("Assets", ", ".join(assets) if assets else "—")
sharpe_saved = selected.get("sharpe_oos")
c4.metric("Sharpe (saved)", f"{sharpe_saved:.2f}" if sharpe_saved is not None else "—")
dd_saved = selected.get("max_drawdown")
c5.metric("Max DD (saved)", f"{dd_saved:.2f}" if dd_saved is not None else "—")

data_sources = selected.get("data_sources") or []
if data_sources:
    st.caption("Data sources: " + ", ".join(f"`{s}`" for s in data_sources))


# ---------- run + render --------------------------------------------------

if not run_clicked:
    st.info(
        "Pick a date range in the sidebar — or hit a regime preset — and press "
        "**Run backtest**. First runs may download data; subsequent runs hit the cache."
    )
    st.stop()

if start >= end:
    st.error("Start date must be before end date.")
    st.stop()

if not symbol:
    st.error("Enter a ticker to run the backtest.")
    st.stop()

with st.spinner(f"Running {selected['name']} on {symbol}, {start} → {end} ..."):
    try:
        out = run(selected, str(start), str(end), symbol=symbol)
    except ValueError as e:
        st.error(f"Couldn't run the backtest: {e}")
        st.stop()
    except FileNotFoundError as e:
        st.error(
            f"Missing data: {e}. The first run for this model may need to "
            "download data from a remote source — try again in a moment."
        )
        st.stop()
    except Exception as e:  # surface any other failure to the user
        st.error(f"Unexpected failure: {type(e).__name__}: {e}")
        st.stop()

data_details.render(selected, out)
data_view.render(selected, out)
results_view.render(out)
options_view.render(out)  # no-op unless the model returned payoff/simulation
