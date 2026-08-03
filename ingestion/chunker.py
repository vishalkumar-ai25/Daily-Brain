"""
ingestion/chunker.py
────────────────────
Layer 1 — Step 3: Cleaned text → list of overlapping text chunks.

Design decision — why fixed-size chunking with overlap?
────────────────────────────────────────────────────────
There are three main chunking strategies:

  1. Fixed-size + overlap  ← we use this
     Split text every N words, with M words of overlap between adjacent chunks.
     Simple, predictable, and good enough for a first iteration.

  2. Sentence-aware
     Group complete sentences up to a token limit. Better for readability,
     slightly harder to implement. Worth trying in a later iteration.

  3. Semantic chunking
     Split at topic-change boundaries using embedding similarity. Most
     "intelligent", but needs an embedding model just to chunk — expensive.

We start with (1) because:
  - It's easy to explain and defend.
  - Overlap solves the main failure mode: context spanning a chunk boundary.
  - You can tune CHUNK_SIZE and OVERLAP to experiment and understand the
    trade-off (bigger chunks = more context per chunk, but fewer chunks,
    so retrieval is coarser).

Default parameters
──────────────────
  CHUNK_SIZE = 250 words  — fits comfortably inside MiniLM's 256-token window
  OVERLAP    =  30 words  — enough to carry a sentence across a boundary

Usage
─────
    from ingestion.chunker import chunk_text

    chunks = chunk_text("Long article text goes here...")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {len(chunk.split())} words")
"""

CHUNK_SIZE: int = 250    # words per chunk
OVERLAP: int = 30        # words of overlap between consecutive chunks


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[str]:
    """
    Split cleaned text into overlapping fixed-size chunks.

    Parameters
    ----------
    text : str
        Cleaned text to chunk (output of cleaner.clean_text).
    chunk_size : int
        Number of words per chunk. Default: 250.
    overlap : int
        Number of words shared between adjacent chunks. Default: 30.
        Must be less than chunk_size.

    Returns
    -------
    list[str]
        List of chunk strings. Each chunk is a space-joined slice of words.
        The list is empty if the input text is empty.

    Example
    -------
    If the text is 600 words long, chunk_size=250, overlap=30:
      Chunk 0: words 0–249
      Chunk 1: words 220–469   (starts 30 words before end of chunk 0)
      Chunk 2: words 440–600   (last chunk may be shorter)
    """
    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap   # how many words we advance each iteration
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break                  # reached the end — stop
        start += step

    return chunks


# ── Manual test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Generate a sample 700-word text for testing
    sample_words = ["word"] * 700
    sample_text = " ".join(sample_words)

    chunks = chunk_text(sample_text)
    print(f"Input: {len(sample_text.split())} words")
    print(f"Chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: {len(c.split())} words")
