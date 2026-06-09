"""EDGAR filings loader: recent-filings index + plain-text filing bodies.

``get_recent_filings`` reads the official submissions API
(``data.sec.gov/submissions/CIK##########.json``), snapshot-cached per day.
``get_filing_text`` fetches a filing's primary document from the Archives,
strips HTML to plain text and caches it forever keyed by accession number —
filings are immutable, so that cache never goes stale.
"""

from __future__ import annotations

import html as html_lib
import json
import re
from typing import cast

import pandas as pd

from tradinglib.data.paths import processed_dir
from tradinglib.loaders.edgar_client import EdgarClient

SOURCE = "filings"
_SUBDIR = "edgar"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{doc}"
_DEFAULT_FORMS = ("8-K", "10-Q", "10-K")

_default_client: EdgarClient | None = None


def _get_client(client: EdgarClient | None) -> EdgarClient:
    global _default_client
    if client is not None:
        return client
    if _default_client is None:
        _default_client = EdgarClient()
    return _default_client


def _strip_html(html: str) -> str:
    """Plain text from a filing document — stdlib only, good enough for excerpts."""
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def get_recent_filings(
    cik: int,
    *,
    forms: tuple[str, ...] = _DEFAULT_FORMS,
    days: int = 120,
    asof: pd.Timestamp | None = None,
    refresh: bool = False,
    client: EdgarClient | None = None,
) -> pd.DataFrame:
    """Recent filings for one company, newest first.

    Filtered to ``forms`` filed within ``days`` of ``asof`` (default: now).
    Schema: ``[cik, form, accession, filing_date, primary_document]``.
    """
    asof = pd.Timestamp.now("UTC") if asof is None else asof
    snapshot = asof.strftime("%Y-%m-%d")
    out = processed_dir(SOURCE) / _SUBDIR / str(cik) / f"submissions-{snapshot}.parquet"
    if out.exists() and not refresh:
        df = pd.read_parquet(out)
    else:
        response = _get_client(client).get(_SUBMISSIONS_URL.format(cik=cik))
        recent = json.loads(response.text)["filings"]["recent"]
        df = pd.DataFrame(
            {
                "cik": cik,
                "form": recent["form"],
                "accession": recent["accessionNumber"],
                "filing_date": pd.to_datetime(recent["filingDate"], utc=True),
                "primary_document": recent["primaryDocument"],
            }
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)

    df["filing_date"] = pd.to_datetime(df["filing_date"], utc=True)
    mask = df["form"].isin(forms) & (df["filing_date"] >= asof - pd.Timedelta(days=days))
    result = df[mask].sort_values("filing_date", ascending=False).reset_index(drop=True)
    return cast(pd.DataFrame, result)


def get_filing_text(
    cik: int,
    accession: str,
    primary_document: str,
    *,
    max_chars: int = 20_000,
    client: EdgarClient | None = None,
) -> str:
    """Plain text of one filing's primary document, truncated to ``max_chars``."""
    cache = processed_dir(SOURCE) / _SUBDIR / str(cik) / f"{accession}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")[:max_chars]

    url = _ARCHIVE_URL.format(
        cik=cik, accession_nodash=accession.replace("-", ""), doc=primary_document
    )
    text = _strip_html(_get_client(client).get(url).text)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text[:max_chars]
