"""
agents/contradiction_finder.py
───────────────────────────────
Layer 3 — Agent 2: Find conflicting claims across entries on the same topic.

This is a multi-step agentic pattern:
  Step 1: Similarity retrieval — find the most relevant chunks for a topic
  Step 2: Group chunks by entry
  Step 3: Call the LLM to compare entries pairwise and flag contradictions

Why this demonstrates agentic reasoning
────────────────────────────────────────
A simple RAG call asks "what does my knowledge base say about X?" and gets
one answer. This agent asks "do different entries say *conflicting* things
about X?" — it runs a retrieval, then does structured multi-entry comparison
using the LLM, then interprets the result. It "plans" (retrieve → compare
→ report) rather than just passing retrieved chunks directly to a generator.

Usage
─────
    from agents.contradiction_finder import find_contradictions

    result = find_contradictions("transformer attention mechanisms")
    print(result)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.embed import query_similar
from db.schema import get_connection
from retrieval.rag import call_llm


def find_contradictions(topic: str, top_k: int = 10) -> str:
    """
    Find potentially contradictory claims in your notes on a topic.

    Parameters
    ----------
    topic : str
        The topic to investigate (e.g., "transformer attention").
    top_k : int
        Number of chunks to retrieve. Higher = more coverage but longer prompt.

    Returns
    -------
    str
        LLM analysis of agreements, disagreements, and nuances across entries.
    """
    # Step 1: retrieve relevant chunks
    chunks = query_similar(topic, top_k=top_k)

    if not chunks:
        return "No relevant entries found for this topic."

    # Step 2: group chunks by entry_id so the LLM sees entry-level context
    entries: dict[int, list[str]] = {}
    for chunk in chunks:
        eid = chunk["entry_id"]
        entries.setdefault(eid, []).append(chunk["chunk_text"])

    # Fetch entry metadata (date, source_type) to give LLM more context
    conn = get_connection()
    cursor = conn.cursor()
    entry_ids = list(entries.keys())
    placeholders = ",".join("?" for _ in entry_ids)
    cursor.execute(
        f"SELECT id, date_added, source_type FROM entries WHERE id IN ({placeholders})",
        entry_ids,
    )
    meta = {row["id"]: row for row in cursor.fetchall()}
    conn.close()

    # Step 3: build comparison prompt
    entry_blocks = []
    for eid, chunks_text in entries.items():
        info = meta.get(eid, {})
        date = (info.get("date_added") or "unknown")[:10]
        source = info.get("source_type", "unknown")
        combined = " ".join(chunks_text)
        entry_blocks.append(f"[Entry #{eid} — {date} — {source}]\n{combined}")

    context = "\n\n---\n\n".join(entry_blocks)

    prompt = f"""You are analysing a personal reading log for the topic: "{topic}"

Here are the relevant passages from multiple entries:

{context}

Please:
1. Identify any CONTRADICTIONS or conflicting claims between entries
   (e.g., Entry #3 says X, but Entry #7 says Y which contradicts X)
2. Identify any AGREEMENTS — claims that multiple entries corroborate
3. Identify NUANCES — cases where entries seem to disagree but are actually
   talking about different contexts or time periods
4. Give an overall assessment: are the sources generally consistent or
   significantly contradictory on this topic?

If there are no real contradictions, say so clearly — don't invent them."""

    return call_llm(prompt)


# ── Manual test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(find_contradictions("machine learning training techniques"))
