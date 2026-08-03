"""
ingestion/scraper.py
────────────────────
Layer 1 — Step 1: URL → raw text extraction.

Takes a URL and returns the main article text (stripping ads, nav bars,
cookie banners, etc.) using the `trafilatura` library.

Why trafilatura?
    It uses heuristics trained on thousands of web pages to identify "main
    content" vs boilerplate. Much more reliable than BeautifulSoup alone.

Usage
─────
    from ingestion.scraper import fetch_url

    text = fetch_url("https://example.com/some-article")
    print(text[:500])
"""

import trafilatura
import requests


def fetch_url(url: str) -> str:
    """
    Fetch a URL and extract the main article text.

    Parameters
    ----------
    url : str
        The URL to fetch.

    Returns
    -------
    str
        The extracted main text, or an empty string if extraction fails.

    Raises
    ------
    ValueError
        If the URL is empty or obviously malformed.
    """
    if not url or not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL: {url!r}. Must start with http:// or https://")

    # Download the page
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        print(f"⚠️  Could not download: {url}")
        return ""

    # Extract main content
    text = trafilatura.extract(
        downloaded,
        include_comments=False,      # skip comment sections
        include_tables=True,         # keep tables (useful for papers/articles)
        no_fallback=False,           # try fallback extraction if main heuristic fails
    )

    if not text:
        print(f"⚠️  Extraction returned empty text for: {url}")
        return ""

    return text


# ── Manual test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    result = fetch_url(test_url)
    print(f"Extracted {len(result.split())} words\n")
    print(result[:800])
