# `_native` — compiled extension escape hatch

This directory is reserved for compiled extensions if and when a specific
model needs them. Empty by default — see [`docs/latency-notes.md`](../../docs/latency-notes.md)
for when this is warranted and when it isn't.

**Default: stay in Python.** Reach for a compiled extension only after
profiling identifies a specific hot path that cannot be sufficiently sped up
by Numba, vectorization, or `torch.compile`.

When that bar is met, the convention is:

- One subdirectory per extension (e.g. `_native/orderbook/`).
- pybind11 (C++) or PyO3 (Rust) bindings exposed as a regular Python module.
- A pure-Python fallback exists for the same function so non-developers can
  still run the notebooks without a compile toolchain.
