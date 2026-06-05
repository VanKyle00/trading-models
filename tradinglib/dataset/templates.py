"""Hand-authored scenario inputs (data only, no logic).

Categories mirror spec §4's dataset composition: explain ~50%, counterfactual
~25%, methodology ~15%, refusal ~10%. Placeholders are filled by scenarios.py.
"""

from __future__ import annotations

CATEGORIES = ("explain", "counterfactual", "methodology", "refusal")

# Free-ticker (classical) models sample from this basket; fixed/choice models
# use their own declared tickers instead.
TICKER_BASKET = ("SPY", "QQQ", "AAPL", "MSFT", "GLD", "IWM")

# Conservative per-family backtest windows. Models whose data source can't serve
# a window simply fail the run and get dropped by the grounding filter; the
# orchestrator logs the drop rate so windows can be tuned.
WINDOWS = {
    "classical": ("2015-01-01", "2024-12-31"),
    "ml": ("2015-01-01", "2024-12-31"),
    "microstructure": ("2024-01-01", "2024-03-31"),
    "alt-data": ("2020-01-01", "2023-12-31"),
    "options": ("2020-01-01", "2023-12-31"),
}
DEFAULT_WINDOW = ("2020-01-01", "2023-12-31")

QUESTION_TEMPLATES = {
    "explain": [
        "How did the {model_name} model do on {symbol} from {start} to {end}?",
        "What's the Sharpe and max drawdown for {model_name} on {symbol}?",
        "Summarize the {model_name} backtest on {symbol}. Is the drawdown concerning?",
        "Walk me through the risk/return of {model_name} on {symbol} over {start}-{end}.",
    ],
    "counterfactual": [
        "Run the {model_name} model on {symbol} instead and tell me if it's better than its default.",
        "For {model_name}, how does {symbol} compare to {symbol2} over {start}-{end}?",
        "What happens to {model_name} on {symbol} if trading fees are 10 bps?",
    ],
    "methodology": [
        "How are trade fills modelled in these backtests?",
        "What does the Deflated Sharpe ratio mean here and why does it matter?",
        "What's the train/test split for the {model_name} model?",
        "What data source does the {model_name} model use?",
    ],
    "refusal": [
        "What's the live price of Bitcoin right now?",
        "Should I buy {symbol} today?",
        "What will {symbol} do next week?",
        "Give me insider information on {model_name}.",
    ],
}
