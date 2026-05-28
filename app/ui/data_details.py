"""Expandable panel summarizing the data the model just consumed.

Shows the actual window the engine saw (first/last bar, count), the
columns of input data passed downstream, the run parameters, and a small
sample of the rows themselves — useful both as a sanity check and as a
visible answer to "what data is this model using?".
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render(model_meta: dict[str, Any], out: dict[str, Any]) -> None:
    data: pd.DataFrame = out["data"]
    params: dict[str, Any] = out.get("params", {})
    symbol = out.get("symbol", "—")

    sources = model_meta.get("data_sources") or []

    with st.expander("📊 Data details", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Window**")
            st.write(f"Symbol: `{symbol}`")
            st.write(f"Bars: `{len(data):,}`")
            if len(data) > 0:
                st.write(f"Start: `{data.index[0]}`")
                st.write(f"End: `{data.index[-1]}`")

        with c2:
            st.markdown("**Series**")
            for col in data.columns:
                st.write(f"`{col}`")
            if sources:
                st.markdown("**Sources**")
                for s in sources:
                    st.write(f"`{s}`")

        with c3:
            st.markdown("**Run params**")
            if params:
                for k, v in params.items():
                    st.write(f"{k}: `{v}`")
            else:
                st.caption("(none reported)")

        st.markdown("**Sample (first 5 rows)**")
        st.dataframe(data.head(5), use_container_width=True)
        st.markdown("**Sample (last 5 rows)**")
        st.dataframe(data.tail(5), use_container_width=True)
