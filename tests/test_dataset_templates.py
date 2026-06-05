from tradinglib.dataset.templates import CATEGORIES, QUESTION_TEMPLATES, TICKER_BASKET, WINDOWS


def test_all_categories_have_templates():
    assert set(QUESTION_TEMPLATES) == set(CATEGORIES)
    for cat in CATEGORIES:
        assert QUESTION_TEMPLATES[cat], f"no templates for {cat}"


def test_templates_use_only_known_placeholders():
    allowed = {"model_name", "symbol", "symbol2", "start", "end"}
    import string
    for cat, templates in QUESTION_TEMPLATES.items():
        for t in templates:
            fields = {f for _, f, _, _ in string.Formatter().parse(t) if f}
            assert fields <= allowed, f"{cat}: unexpected placeholder in {t!r}: {fields - allowed}"


def test_basket_and_windows_nonempty():
    assert len(TICKER_BASKET) >= 3
    assert all(len(w) == 2 for w in WINDOWS.values())
