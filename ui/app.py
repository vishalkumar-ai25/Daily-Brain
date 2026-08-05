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
    st.success("✅ Layer 2: RAG (Q2)")
    st.success("✅ Layer 3: Agents (Q3)")
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

# Initialize session state keys for persisting fetched article across reruns
if "fetched_text" not in st.session_state:
    st.session_state["fetched_text"] = ""
if "fetched_url" not in st.session_state:
    st.session_state["fetched_url"] = ""

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
                fetched = fetch_url(url_input)
            if fetched:
                # Persist in session state so it survives the next rerun
                st.session_state["fetched_text"] = fetched
                st.session_state["fetched_url"] = url_input
                st.success(f"✅ Fetched {len(fetched.split())} words — click 💾 Save to Daily Brain below")
            else:
                st.error("Could not extract text from this URL. Try pasting manually.")

        # Always restore from session state (survives button-click reruns)
        raw_text = st.session_state["fetched_text"]
        source_url = st.session_state["fetched_url"] or None

        if raw_text:
            word_count = len(raw_text.split())
            st.info(f"📄 **Article in memory:** {word_count} words from `{source_url}`")
            with st.expander("📖 Preview article text"):
                st.text_area("Full article preview", raw_text, height=300, disabled=True)
        else:
            st.caption("Enter a URL above and click Fetch Article.")

    else:
        # Switching to paste mode — clear any previously fetched URL article
        st.session_state["fetched_text"] = ""
        st.session_state["fetched_url"] = ""
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

        # Clear session state after successful save
        st.session_state["fetched_text"] = ""
        st.session_state["fetched_url"] = ""

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

            # ── Show diagrams & images from source entries ──────────────────
            images = result.get("images", [])
            if images:
                st.divider()
                st.subheader(f"🖼️ Diagrams & Images from Source ({len(images)} found)")
                st.caption("These are the original diagrams from the article(s) used to answer your question.")
                for img in images:
                    try:
                        caption = img["alt"] if img["alt"] else f"Image from Entry #{img['entry_id']}"
                        st.image(img["url"], caption=caption, use_container_width=True)
                    except Exception as img_err:
                        st.warning(f"Could not load image: `{img['url'][:80]}...` — {img_err}")

    with st.expander("📖 How RAG & Semantic Search work"):
        st.markdown("""
**1. Vector Search (Semantic Search)**
Converts your text into a 384-dimensional dense vector using `all-MiniLM-L6-v2`. It compares your query vector against stored chunk vectors in Chroma DB using **Cosine Distance**. Lower distance = higher semantic similarity.

**2. Full RAG (Retrieval-Augmented Generation)**
Retrieves top matching chunks, embeds them into a strict prompt instruction ("Answer ONLY using the provided context"), and queries the LLM (Ollama/Gemini) to produce a grounded response with zero hallucinations.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Agents — Layer 3 (FULLY IMPLEMENTED)
# ─────────────────────────────────────────────────────────────────────────────

# Session state: multi-step Q&A agent instance + conversation history
if "msqa_agent" not in st.session_state:
    from agents.multistep_qa import MultiStepQA
    st.session_state["msqa_agent"] = MultiStepQA()
if "msqa_history_display" not in st.session_state:
    st.session_state["msqa_history_display"] = []  # list of {"role", "content"}

with tab_agents:
    st.header("🤖 AI Agents")
    st.caption("Three agentic workflows powered by your knowledge base")

    agent_tab1, agent_tab2, agent_tab3 = st.tabs([
        "📅 Weekly Summary",
        "⚡ Contradiction Finder",
        "💬 Multi-step Q&A",
    ])

    # ── Agent 1: Weekly Summary ───────────────────────────────────────────────
    with agent_tab1:
        st.subheader("📅 Weekly Summary")
        st.markdown(
            "Auto-summarize **everything you read** in a given time window. "
            "Uses date-range retrieval (not similarity search) — a different agentic retrieval pattern."
        )
        st.divider()

        days = st.slider("Look back how many days?", min_value=1, max_value=90, value=7, step=1)

        # Show entry count for that window
        conn = get_connection()
        from datetime import datetime, timedelta
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        count_in_window = conn.execute(
            "SELECT COUNT(*) FROM entries WHERE date_added >= ?", (cutoff,)
        ).fetchone()[0]
        conn.close()

        if count_in_window == 0:
            st.warning(f"No entries found in the last {days} days. Try increasing the range or add more entries.")
        else:
            st.info(f"📚 **{count_in_window} entries** found in the last {days} days.")

        if st.button("✨ Generate Summary", type="primary", disabled=(count_in_window == 0)):
            with st.spinner(f"Reading {count_in_window} entries and summarising..."):
                from agents.weekly_summary import generate_weekly_summary
                summary = generate_weekly_summary(days=days)

            st.success("Summary ready!")
            st.divider()
            st.markdown(summary)

            # Download button
            st.download_button(
                "📥 Download Summary",
                data=summary,
                file_name=f"daily_brain_summary_{days}d.md",
                mime="text/markdown",
            )

    # ── Agent 2: Contradiction Finder ─────────────────────────────────────────
    with agent_tab2:
        st.subheader("⚡ Contradiction Finder")
        st.markdown(
            "Finds **conflicting or agreeing claims** across your entries on the same topic. "
            "Retrieves the top-K relevant chunks, groups them by source, "
            "then asks the LLM to compare them for contradictions and nuances."
        )
        st.divider()

        topic_input = st.text_input(
            "Topic to investigate",
            placeholder="e.g., merge sort, transformer attention, graph traversal",
            key="contradiction_topic",
        )
        top_k_cf = st.slider("Chunks to compare", min_value=4, max_value=20, value=10, step=2)

        if st.button("🔍 Find Contradictions", type="primary", disabled=not topic_input):
            with st.spinner(f"Retrieving {top_k_cf} chunks on '{topic_input}' and analysing..."):
                from agents.contradiction_finder import find_contradictions
                result = find_contradictions(topic_input, top_k=top_k_cf)

            st.success("Analysis complete!")
            st.divider()
            st.markdown(result)

            st.download_button(
                "📥 Download Analysis",
                data=result,
                file_name=f"contradiction_{topic_input[:30].replace(' ', '_')}.md",
                mime="text/markdown",
            )

    # ── Agent 3: Multi-step Q&A ───────────────────────────────────────────────
    with agent_tab3:
        st.subheader("💬 Multi-step Q&A")
        st.markdown(
            "A **conversational agent** that remembers the conversation. "
            "Ask a follow-up question and it will use both fresh context AND your prior conversation to answer."
        )

        agent: "MultiStepQA" = st.session_state["msqa_agent"]

        col_meta1, col_meta2, col_meta3 = st.columns(3)
        col_meta1.metric("Turns this session", agent.turn_count)
        col_meta2.metric("Max history kept", agent.max_history_turns)
        if col_meta3.button("🗑️ Reset Conversation", type="secondary"):
            agent.reset()
            st.session_state["msqa_history_display"] = []
            st.rerun()

        st.divider()

        # Display conversation history
        history_display = st.session_state["msqa_history_display"]
        if not history_display:
            st.caption("No conversation yet. Ask your first question below.")
        else:
            for turn in history_display:
                if turn["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(turn["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(turn["content"])
                        if turn.get("sources"):
                            st.caption(f"📚 Sources: Entries {turn['sources']}")

        # Input box at the bottom
        msqa_question = st.chat_input("Ask a follow-up question...", key="msqa_input")

        if msqa_question:
            # Add user message to display immediately
            st.session_state["msqa_history_display"].append({
                "role": "user",
                "content": msqa_question,
            })
            with st.spinner("Thinking..."):
                result = agent.ask(msqa_question)

            st.session_state["msqa_history_display"].append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
            st.rerun()


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
