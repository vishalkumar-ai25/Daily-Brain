# 🧠 Daily Brain

> A personal AI knowledge assistant built one day at a time — 365 days, four AI layers, one genuinely understood system.

---

## What is Daily Brain?

Daily Brain is a personal AI system that grows with you over a year. Every day (~10 minutes), you feed it one piece of information — an article, a paper abstract, a class note, a tech blog post. Behind that simple daily habit, a real AI pipeline evolves across four quarterly build phases:

| Quarter | Layer | What it does |
|---------|-------|--------------|
| Q1 (Month 1-3) | Ingestion + NLP | Cleans, chunks, embeds, and stores everything you feed it |
| Q2 (Month 4-6) | RAG | Answers questions grounded in your own data |
| Q3 (Month 7-9) | Agents + GenAI | Weekly summaries, contradiction finding, multi-step reasoning |
| Q4 (Month 10-12) | Vision / CV | Understands screenshots and diagrams too |

---

## System Architecture

```
┌─────────────────────────────────────────┐
│ Layer 4: Vision (Q4)                    │  ← understands screenshots/diagrams
├─────────────────────────────────────────┤
│ Layer 3: Agent / GenAI reasoning (Q3)   │  ← summarizes, finds contradictions
├─────────────────────────────────────────┤
│ Layer 2: RAG (Q2)                       │  ← answers questions from your data
├─────────────────────────────────────────┤
│ Layer 1: Ingestion + NLP (Q1)           │  ← cleans, chunks, stores, embeds
├─────────────────────────────────────────┤
│ Foundation: Storage (Month 1)           │  ← the database everything sits on
└─────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Storage | SQLite | Zero-setup, file-based, perfect for a personal project |
| Vector store | Chroma | Free, local, simple persistence |
| Embeddings | sentence-transformers (MiniLM) | Free, fast, CPU-friendly |
| LLM | Ollama (local) or free-tier API | Local = strong differentiator |
| Agent orchestration | Custom Python loop / LangGraph | You already know LangGraph |
| OCR | Tesseract | Free, well-established |
| Vision | Multimodal LLM API | Most modern APIs support image input |
| UI | Streamlit | Fast to build, great for personal tools |

---

## Folder Structure

```
daily-brain/
├── README.md
├── requirements.txt
├── .gitignore
├── db/
│   └── daily_brain.sqlite       # auto-created on first run
├── ingestion/
│   ├── __init__.py
│   ├── scraper.py               # URL → raw text
│   ├── cleaner.py               # text cleaning logic
│   └── chunker.py               # fixed-size chunking with overlap
├── embeddings/
│   ├── __init__.py
│   └── embed.py                 # embedding generation + storage
├── retrieval/
│   ├── __init__.py
│   └── rag.py                   # query → retrieve → generate
├── agents/
│   ├── __init__.py
│   ├── weekly_summary.py        # auto-summarize last 7 days
│   └── contradiction_finder.py  # find conflicting entries
├── vision/
│   ├── __init__.py
│   ├── ocr.py                   # extract text from images
│   └── describe_image.py        # describe diagrams via multimodal LLM
├── ui/
│   └── app.py                   # Streamlit frontend
└── eval/
    └── test_questions.md        # RAG evaluation log
```

---

## Getting Started

### 1. Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/daily-brain.git
cd daily-brain
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Initialize the database

```bash
python3 -c "from db.schema import init_db; init_db()"
```

### 3. Run the UI

```bash
streamlit run ui/app.py
```

---

## Dev Log

See [`devlog.md`](devlog.md) — one entry per build session tracking what was built, changed, and learned.

---

## Evaluation

See [`eval/test_questions.md`](eval/test_questions.md) — 20 test cases tracking retrieval quality and LLM grounding across all quarters.

---

## Build Milestones

- [ ] **Month 1** — SQLite schema, ingestion script, basic UI shell
- [ ] **Month 2** — Cleaning + chunking pipeline
- [ ] **Month 3** — Embedding generation + Chroma vector store
- [ ] **Month 4** — RAG query → retrieve → answer pipeline
- [ ] **Month 5** — Evaluation log + retrieval quality tuning
- [ ] **Month 6** — Citation display + grounding verification
- [ ] **Month 7** — Weekly auto-summary agent
- [ ] **Month 8** — Contradiction finder agent
- [ ] **Month 9** — Multi-step Q&A (multi-retrieval reasoning)
- [ ] **Month 10** — OCR pipeline (Tesseract)
- [ ] **Month 11** — Diagram description (multimodal LLM)
- [ ] **Month 12** — Unified text+image retrieval, final polish

---

*Built slowly. Understood deeply. Defensible completely.*
