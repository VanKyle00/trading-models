# Latency notes — when does C++ matter?

A quick reference for the question "do we need to leave Python for this?"
Short version: **for training, never. For inference, only at sub-millisecond
latencies, and only if you have venue access that justifies the engineering
investment.** Most of what looks like a latency problem is solvable inside
Python.

## Training is always Python

Every serious quant shop — Citadel, Jane Street, Two Sigma, Renaissance —
trains models in Python (PyTorch, JAX, XGBoost, scikit-learn). Training
speed is not the bottleneck for portfolio research; iteration speed is.
Reaching for C++ here trades developer productivity for compute that you
already have plenty of.

## Inference: it depends on the latency budget

| Latency budget | What it looks like | Language |
| --- | --- | --- |
| Sub-µs (true HFT) | Colocated at the exchange; FPGAs; kernel bypass; tick-to-trade in nanoseconds | C++ on bare metal, plus FPGAs |
| Sub-ms | Market-making, fast quoting strategies | C++ usually; Numba / Cython / `torch.compile` can come close |
| 10–100 ms | Most "low-latency" intraday strategies | Python, written carefully |
| Seconds+ | Swing, end-of-day, alt-data, most ML | Python — overkill-fast |

For a portfolio repo without colocation or institutional venue access, the
realistic ceiling is the "10–100 ms" tier. There is no reason to leave
Python for this work.

## When a model genuinely *does* need a speedup

The right escalation path inside Python, in order:

1. **Vectorize.** Replace per-bar Python loops with NumPy / Polars
   operations. Often a 100x speedup with no exotic tools.
2. **Numba.** Annotate the hot function with `@numba.jit(nopython=True)`.
   Works on a NumPy-flavored subset of Python; gets you close to C speed
   for numeric kernels.
3. **`torch.compile` / TorchScript.** If the hot path is a neural-network
   inference call.
4. **Cython.** If you need fine-grained control and don't want to bring in
   a build toolchain.
5. **Compiled extension via pybind11 (C++) or PyO3 (Rust).** Reserved for
   when 1–4 have been benchmarked and found insufficient.

Conventions for step 5 are documented in
[`tradinglib/_native/README.md`](../tradinglib/_native/README.md). The
directory is empty by default; any addition should ship with a pure-Python
fallback so the notebooks still run for someone who doesn't have a C++
toolchain installed.

## How to demonstrate latency awareness without rewriting everything

In a portfolio context, the strongest signal is not "I rewrote the
backtest engine in C++" — it's "I understand precisely *where* C++ would
matter and have profiled the hot paths to prove it."

A focused benchmark notebook under `notebooks/eda/` that does the
following demonstrates this clearly:

1. Pick one hot path (e.g., reconstructing an L2 order book from a tick
   stream, or computing realized volatility on a long horizon).
2. Implement it in pure Python (vectorized NumPy is the baseline).
3. Implement the same thing with Numba.
4. Implement it as a minimal pybind11 (or PyO3) extension.
5. Plot the three on a log-scale latency chart with a brief discussion of
   when each tier is the right choice.

This is the "I understand microstructure latency" milestone for the repo,
and it lives alongside the microstructure model family rather than driving
the whole codebase into C++.

## What this repo does *not* do

- Optimize for sub-millisecond inference.
- Maintain a C++ build toolchain as a default requirement.
- Compile any extensions in CI.

If a specific model in the future genuinely needs an extension, that
model's directory can ship the build setup and an `_native/` extension
that the rest of the repo ignores.
