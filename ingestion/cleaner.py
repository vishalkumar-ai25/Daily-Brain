"""
ingestion/cleaner.py
────────────────────
Layer 1 — Step 2: Raw text → cleaned text.

What "cleaning" means here:
  - Strip leading/trailing whitespace
  - Collapse multiple consecutive blank lines to one
  - Remove broken unicode replacement characters (U+FFFD)
  - Remove obviously junk lines (cookie banners, "subscribe now", etc.)

What we deliberately do NOT do:
  - Lowercase — our embedding model (MiniLM) is cased; lowercasing hurts quality
  - Stemming / lemmatisation — we're not building a keyword index; the embedding
    model handles semantic meaning on its own

Usage
─────
    from ingestion.cleaner import clean_text

    cleaned = clean_text(raw_text)
"""

import re
import unicodedata


# Common web boilerplate phrases to strip (line-level matches)
_JUNK_PATTERNS: list[re.Pattern] = [
    re.compile(r"subscribe\s+now", re.IGNORECASE),
    re.compile(r"sign\s+up\s+(for\s+)?our\s+newsletter", re.IGNORECASE),
    re.compile(r"cookie\s+policy", re.IGNORECASE),
    re.compile(r"accept\s+all\s+cookies", re.IGNORECASE),
    re.compile(r"^\s*advertisement\s*$", re.IGNORECASE),
    re.compile(r"^\s*share\s+this\s+(article|post|story)\s*$", re.IGNORECASE),
]


def _is_junk_line(line: str) -> bool:
    """Return True if the line matches any known junk pattern."""
    return any(p.search(line) for p in _JUNK_PATTERNS)


def clean_text(raw: str) -> str:
    """
    Clean raw text for ingestion into Daily Brain.

    Parameters
    ----------
    raw : str
        The raw text (from scraper or pasted directly).

    Returns
    -------
    str
        Cleaned text, ready for chunking.
    """
    if not raw or not raw.strip():
        return ""

    # 1. Normalise unicode — convert fancy quotes, dashes, etc. to ASCII equivalents
    #    NFC normalisation makes sure é is stored as one character, not e + combining accent
    text = unicodedata.normalize("NFC", raw)

    # 2. Remove unicode replacement character (appears when encoding is broken)
    text = text.replace("\ufffd", "")

    # 3. Normalise Windows-style line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Filter out junk lines
    lines = text.split("\n")
    lines = [line for line in lines if not _is_junk_line(line)]

    # 5. Collapse 3+ consecutive blank lines into 2 (preserve paragraph breaks)
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    # 6. Strip each line's trailing whitespace
    cleaned_lines = [line.rstrip() for line in cleaned_lines]

    # 7. Rejoin and strip overall
    result = "\n".join(cleaned_lines).strip()

    return result


# ── Manual test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    This is a real article paragraph. It talks about something interesting.

    Subscribe now to get our newsletter delivered to your inbox!

    Another real paragraph here, with some content.


    Accept all cookies to continue browsing.

    Final paragraph of actual content.
    """
    print(clean_text(sample))
