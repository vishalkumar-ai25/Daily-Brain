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
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def fetch_url(url: str) -> str:
    """
    Fetch a URL and extract the main article text along with diagrams and images.

    Parameters
    ----------
    url : str
        The URL to fetch.

    Returns
    -------
    str
        The extracted main text with markdown image links for diagrams.
    """
    if not url or not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL: {url!r}. Must start with http:// or https://")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        downloaded = resp.text
    except Exception as e:
        print(f"⚠️  Direct request failed for {url}: {e}, falling back to trafilatura download")
        downloaded = trafilatura.fetch_url(url)

    if not downloaded:
        print(f"⚠️  Could not download: {url}")
        return ""

    # Extract main content text
    text = trafilatura.extract(
        downloaded,
        include_comments=False,      # skip comment sections
        include_tables=True,         # keep tables (useful for papers/articles)
        no_fallback=False,           # try fallback extraction if main heuristic fails
    ) or ""

    # Extract diagrams and content images via BeautifulSoup
    try:
        soup = BeautifulSoup(downloaded, "html.parser")
        article = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=lambda c: c and any(k in c.lower() for k in ["content", "article", "post", "entry"]))
            or soup
        )

        images_found: list[str] = []
        for img in article.find_all("img"):
            src = img.get("src") or img.get("data-src") or (img.get("srcset", "").split()[0] if img.get("srcset") else None)
            if not src:
                continue
            alt = img.get("alt", "").strip() or img.get("title", "").strip() or "Article Diagram"

            # Filter out UI icons, site logos, tracking buttons, location SVGs
            src_lower = src.lower()
            alt_lower = alt.lower()
            if any(ignore in src_lower or ignore in alt_lower for ignore in [".svg", "logo", "icon", "location", "arrow", "avatar", "profile", "button", "ad-", "footer"]):
                continue

            if not src.startswith("http"):
                src = urljoin(url, src)

            md_img = f"![Diagram: {alt}]({src})"
            if md_img not in images_found:
                images_found.append(md_img)

        if images_found:
            text += "\n\n### 🖼️ Article Diagrams & Visuals\n" + "\n\n".join(images_found)
    except Exception as err:
        print(f"⚠️  Image tag extraction warning: {err}")

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
