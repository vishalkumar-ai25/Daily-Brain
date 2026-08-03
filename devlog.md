# Dev Log — Daily Brain

One entry per build session. This is your honest record of what you built,
what you changed, and what you understood (or didn't).

Format each entry as:

```
## [Date] — Session N
**Built:** what file/feature was created
**Changed:** what you modified from the generated scaffold and why
**Understood:** the concept you can now explain without looking anything up
**Still fuzzy on:** what you'd go back and ask about
```

---

## 2026-08-03 — Session 1

**Built:**
- Project scaffold: db/, ingestion/, embeddings/, retrieval/, agents/, vision/, ui/, eval/
- db/schema.py — SQLite schema with `entries` and `chunks` tables
- ingestion/scraper.py — URL → text via trafilatura
- ingestion/cleaner.py — text cleaning (junk line removal, unicode normalisation)
- ingestion/chunker.py — fixed-size chunking with overlap
- embeddings/embed.py — MiniLM embeddings + Chroma vector store
- retrieval/rag.py — full RAG pipeline (stub, Layer 2)
- agents/weekly_summary.py — date-range retrieval + LLM summary (stub, Layer 3)
- agents/contradiction_finder.py — multi-entry comparison agent (stub, Layer 3)
- vision/ocr.py — Tesseract OCR (stub, Layer 4)
- vision/describe_image.py — multimodal LLM description (stub, Layer 4)
- ui/app.py — Streamlit frontend with all 5 tabs

**Changed:**
- (fill in: anything you modified from the generated code and why)

**Understood:**
- (fill in: the concept you can now explain unaided — e.g., "what an embedding is")

**Still fuzzy on:**
- (fill in: what you'd go back and ask about before moving on)

---

*(add a new entry each build session)*
