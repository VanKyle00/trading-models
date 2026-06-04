# tests/test_webapp_events.py
from webapp.events import events_for_assets


def test_equities_presets_include_covid_and_gfc():
    labels = [e["label"] for e in events_for_assets(("equities",))]
    assert "COVID Crash" in labels
    assert "GFC 2008" in labels


def test_crypto_presets_include_ftx_not_gfc():
    labels = [e["label"] for e in events_for_assets(("crypto",))]
    assert "FTX Collapse" in labels
    assert "GFC 2008" not in labels  # equities-only event


def test_unknown_asset_class_returns_empty():
    assert events_for_assets(("forex",)) == []
    assert events_for_assets(()) == []


def test_presets_are_iso_dated_and_ordered():
    for assets in (("equities",), ("crypto",)):
        for e in events_for_assets(assets):
            assert e["start"] < e["end"]  # ISO strings sort chronologically
            assert len(e["start"]) == 10 and len(e["end"]) == 10
