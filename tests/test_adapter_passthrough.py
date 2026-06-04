"""The GUI adapter forwards extra kwargs to a model's run_for_gui."""

from __future__ import annotations

from app import adapters


def test_run_forwards_extra_params(monkeypatch) -> None:
    captured = {}

    def fake_loader(model_dir):
        class M:
            @staticmethod
            def run_for_gui(start, end, **params):
                captured.update(params)
                return {"ok": True}
        return M

    monkeypatch.setattr(adapters, "_load_backtest_module", fake_loader)
    out = adapters.run({"_path": "models/options/01-delta-hedged-long-option-spy"},
                       "2023-01-01", "2023-02-01",
                       symbol="SPY", implied_vol=0.25, tenor_days=21, n_paths=500)
    assert out == {"ok": True}
    assert captured == {"symbol": "SPY", "implied_vol": 0.25, "tenor_days": 21, "n_paths": 500}
