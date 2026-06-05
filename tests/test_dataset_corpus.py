from tradinglib.dataset.corpus import Corpus, chunk_text, discover_docs


def test_discover_finds_methodology_and_model_cards():
    paths = discover_docs()
    names = {p.name for p in paths}
    assert "methodology.md" in names
    assert any(p.name == "model.md" for p in paths)


def test_chunk_text_strips_html_and_splits():
    chunks = chunk_text("<h1>Fills</h1><p>Next-open execution.</p>" * 50, max_chars=120)
    assert len(chunks) > 1
    assert "<h1>" not in chunks[0]


def test_corpus_search_returns_relevant_chunk():
    corpus = Corpus.from_chunks(
        [
            "Trade fills use next-open execution to avoid look-ahead bias.",
            "The Sharpe ratio annualizes mean return over volatility.",
        ]
    )
    hits = corpus.search("how are fills modelled?", k=1)
    assert hits and "next-open" in hits[0]
