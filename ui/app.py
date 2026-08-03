"""
ui/app.py
─────────
Daily Brain — Streamlit UI

The single front-end for all four layers. Tabs are shown/hidden based on
what's actually implemented — so this file can live from Day 1 of the project
even though Layer 2-4 features are stubs until you build them.

Run:
    streamlit run ui/app.py
"""

import sys
from pathlib import Path
import streamlit as st
import sqlite3

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.schema import init_db, get_connection
from ingestion.scraper import fetch_url
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Daily Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialise DB (idempotent) ────────────────────────────────────────────────
init_db()


# ── Helper: save entry to DB ──────────────────────────────────────────────────
def save_entry(source_type: str, raw_text: str, cleaned_text: str, source_url: str | None) -> int:
    """Insert a new entry into the database and return its ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO entries (source_type, raw_text, cleaned_text, source_url)
           VALUES (?, ?, ?, ?)""",
        (source_type, raw_text, cleaned_text, source_url),
    )
    entry_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return entry_id


def save_chunks(entry_id: int, chunks: list[str]) -> None:
    """Insert chunks for an entry (without embeddings yet — Layer 1 month 2-3)."""
    conn = get_connection()
    cursor = conn.cursor()
    for chunk in chunks:
        cursor.execute(
            "INSERT INTO chunks (entry_id, chunk_text) VALUES (?, ?)",
            (entry_id, chunk),
        )
    conn.commit()
    conn.close()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Daily Brain")
    st.caption("Your personal AI knowledge assistant")
    st.divider()

    # Entry count
    conn = get_connection()
    total_entries = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()

    st.metric("Total entries", total_entries)
    st.metric("Total chunks", total_chunks)
    st.divider()

    st.caption("**Layers built so far:**")
    st.success("✅ Layer 1: Ingestion + NLP")
    st.info("🔜 Layer 2: RAG (Q2)")
    st.info("🔜 Layer 3: Agents (Q3)")
    st.info("🔜 Layer 4: Vision (Q4)")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_add, tab_browse, tab_ask, tab_agents, tab_vision = st.tabs([
    "➕ Add Entry",
    "📚 Browse",
    "💬 Ask (RAG)",
    "🤖 Agents",
    "🖼️ Vision",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Add Entry
# ─────────────────────────────────────────────────────────────────────────────
with tab_add:
    st.header("Add Today's Entry")
    st.caption("Feed Daily Brain with something you actually read today (~10 min/day habit)")

    input_mode = st.radio(
        "Input type",
        ["🌐 URL (fetch article)", "📝 Paste text"],
        horizontal=True,
    )

    source_type = st.selectbox(
        "Source type",
        ["article", "paper", "note", "screenshot"],
        help="What kind of thing are you adding?",
    )

    raw_text = ""
    source_url = None

    if input_mode == "🌐 URL (fetch article)":
        url_input = st.text_input("Article URL", placeholder="https://...")
        if st.button("Fetch Article", type="primary") and url_input:
            with st.spinner("Fetching and extracting article text..."):
                raw_text = fetch_url(url_input)
                source_url = url_input
            if raw_text:
                st.success(f"Fetched {len(raw_text.split())} words")
                st.text_area("Preview (first 500 chars)", raw_text[:500], height=120, disabled=True)
            else:
                st.error("Could not extract text from this URL. Try pasting manually.")
    else:
        raw_text = st.text_area(
            "Paste your text here",
            height=300,
            placeholder="Paste the article, abstract, or notes you read today...",
        )

    if raw_text and st.button("💾 Save to Daily Brain", type="primary"):
        with st.spinner("Cleaning, chunking, and embedding..."):
            cleaned = clean_text(raw_text)
            chunks = chunk_text(cleaned)
            entry_id = save_entry(source_type, raw_text, cleaned, source_url)
            save_chunks(entry_id, chunks)
            
            # Generate embeddings and store in Chroma DB
            from embeddings.embed import add_chunks_for_entry
            add_chunks_for_entry(entry_id=entry_id, chunks=chunks)

        st.success(f"✅ Saved & Embedded! Entry #{entry_id} — {len(chunks)} chunks indexed.")

        with st.expander("View chunks"):
            for i, c in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}** ({len(c.split())} words):")
                st.text(c)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Browse
# ─────────────────────────────────────────────────────────────────────────────
with tab_browse:
    st.header("Browse Your Knowledge Base")

    conn = get_connection()
    entries = conn.execute(
        "SELECT id, date_added, source_type, source_url, raw_text, cleaned_text FROM entries ORDER BY date_added DESC"
    ).fetchall()
    conn.close()

    if not entries:
        st.info("No entries yet. Add your first entry in the '➕ Add Entry' tab!")
    else:
        for entry in entries:
            date_str = entry["date_added"][:10] if entry["date_added"] else "unknown"
            badge = {"article": "📰", "paper": "📄", "note": "📝", "screenshot": "🖼️"}.get(
                entry["source_type"], "📌"
            )
            
            # Use first line or first 80 chars as summary header
            first_line = (entry["cleaned_text"] or "").strip().split("\n")[0][:80]
            header_title = f"{badge} Entry #{entry['id']} — {date_str} — {first_line}..."

            with st.expander(header_title):
                if entry['source_url']:
                    st.markdown(f"🔗 **Source URL:** [{entry['source_url']}]({entry['source_url']})")
                
                st.markdown(f"**Type:** `{entry['source_type']}` | **Date Added:** `{date_str}`")
                st.divider()
                st.markdown("### Full Content")
                st.markdown(entry["cleaned_text"] or "*No cleaned text*")
                
                # Fetch chunks for this entry
                conn = get_connection()
                chunks = conn.execute(
                    "SELECT chunk_text FROM chunks WHERE entry_id = ?", (entry["id"],)
                ).fetchall()
                conn.close()
                
                if chunks:
                    with st.expander(f"📦 View {len(chunks)} text chunks"):
                        for i, chk in enumerate(chunks):
                            st.caption(f"Chunk {i+1}:")
                            st.code(chk["chunk_text"], language="text")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Ask (RAG) — Layer 2 stub
# ─────────────────────────────────────────────────────────────────────────────
with tab_ask:
    st.header("Ask Your Knowledge Base")
    st.caption("Search vector embeddings or get AI answers grounded in your notes")

    search_query = st.text_input("Ask a question or search concepts...", placeholder="e.g. How does merge sort count inversions?")
    
    col_search, col_rag = st.columns(2)
    
    with col_search:
        do_search = st.button("🔍 Vector Search (Semantic)", type="secondary")
    with col_rag:
        do_rag = st.button("💬 Ask LLM (Full RAG)", type="primary")

    if search_query:
        if do_search:
            with st.spinner("Embedding query & searching Chroma DB..."):
                from embeddings.embed import query_similar
                results = query_similar(search_query, top_k=5)
            
            if not results:
                st.warning("No matching vector chunks found.")
            else:
                st.subheader("Top Matching Chunks")
                for i, r in enumerate(results):
                    similarity_pct = max(0, int((1 - r["distance"]) * 100))
                    st.markdown(f"**Result {i+1}** (Entry #{r['entry_id']} — Relevancy ~{similarity_pct}% / Distance `{r['distance']:.4f}`) :")
                    st.code(r["chunk_text"], language="text")

        elif do_rag:
            with st.spinner("Retrieving context & generating grounded answer..."):
                import importlib
                import retrieval.rag
                importlib.reload(retrieval.rag)
                from retrieval.rag import answer_question
                result = answer_question(search_query)
            
            st.subheader("🤖 Answer")
            st.markdown(result["answer"])
            if result["sources"]:
                st.info(f"**Sources used:** Entries {result['sources']}")

    with st.expander("📖 How RAG & Semantic Search work"):
        st.markdown("""
**1. Vector Search (Semantic Search)**
Converts your text into a 384-dimensional dense vector using `all-MiniLM-L6-v2`. It compares your query vector against stored chunk vectors in Chroma DB using **Cosine Distance**. Lower distance = higher semantic similarity.

**2. Full RAG (Retrieval-Augmented Generation)**
Retrieves top matching chunks, embeds them into a strict prompt instruction ("Answer ONLY using the provided context"), and queries the LLM (Ollama/Gemini) to produce a grounded response with zero hallucinations.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Agents — Layer 3 stub
# ─────────────────────────────────────────────────────────────────────────────
with tab_agents:
    st.header("AI Agents")
    st.info("🔜 **Coming in Q3 (Month 7-9):** Weekly summaries, contradiction finding, and multi-step Q&A.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Weekly Summary")
        st.caption("Auto-summarize everything you read this week")
        st.button("Generate Summary", disabled=True)

    with col2:
        st.subheader("⚡ Contradiction Finder")
        st.caption("Find conflicting claims across your entries")
        topic = st.text_input("Topic to investigate", disabled=True, placeholder="e.g., transformer attention")
        st.button("Find Contradictions", disabled=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: Vision — Layer 4 stub
# ─────────────────────────────────────────────────────────────────────────────
with tab_vision:
    st.header("Vision / Image Understanding")
    st.info("🔜 **Coming in Q4 (Month 10-12):** Upload screenshots and diagrams — the system will extract and understand their content.")

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], disabled=True)

    with st.expander("🏛️ Architectural decision: why convert images to text?"):
        st.markdown("""
Instead of building a separate image-embedding pipeline, Daily Brain converts images to text first:
- **OCR** (Tesseract) for text-heavy images (screenshots, scanned notes)
- **Vision LLM** (Gemini/GPT-4V) for diagrams and figures

Once converted, image-derived text flows through the **exact same** clean → chunk → embed → store pipeline as regular text entries.

**Result:** one unified retrieval system serves both text and image-derived content. No changes needed to the RAG or agent layers.

**Trade-off:** we lose some visual nuance that a native image embedding might capture — but for a personal knowledge assistant, this simplicity is clearly worth it.
        """)
