"""
agents/weekly_summary.py
────────────────────────
Layer 3 — Agent 1: Auto-summarize everything you read in the last 7 days.

This is an agentic pattern — not a single RAG call. It:
  1. Queries the database for all entries from the last N days (a date-range
     SQL query, not a similarity search — a different retrieval pattern)
  2. Concatenates their text
  3. Calls the LLM to produce a structured weekly summary

Why this is different from a plain RAG call
────────────────────────────────────────────
A RAG call embeds a question and retrieves the *most similar* chunks. This
agent retrieves everything from a *time window* regardless of similarity —
a completely different retrieval axis. Knowing when to use each is what
makes you sound senior in an interview.

Usage
─────
    from agents.weekly_summary import generate_weekly_summary

    summary = generate_weekly_summary(days=7)
    print(summary)
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.schema import get_connection
from retrieval.rag import call_llm


def generate_weekly_summary(days: int = 7) -> str:
    """
    Generate a structured summary of all entries added in the last `days` days.

    Parameters
    ----------
    days : int
        How many days back to look. Default: 7.

    Returns
    -------
    str
        A formatted summary from the LLM.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cutoff = datetime.utcnow() - timedelta(days=days)

    # Retrieve entries by date range — NOT similarity search
    cursor.execute("""
        SELECT id, date_added, source_type, cleaned_text, source_url
        FROM entries
        WHERE date_added >= ?
        ORDER BY date_added ASC
    """, (cutoff.isoformat(),))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return f"No entries found in the last {days} days. Keep feeding Daily Brain!"

    # Build a summary context block
    entry_blocks = []
    for row in rows:
        date_str = row["date_added"][:10]  # YYYY-MM-DD
        source = row["source_type"]
        text_preview = (row["cleaned_text"] or "")[:400]  # first 400 chars
        url = f"\n   URL: {row['source_url']}" if row["source_url"] else ""
        entry_blocks.append(
            f"[Entry #{row['id']} — {date_str} — {source}]{url}\n{text_preview}..."
        )

    context = "\n\n".join(entry_blocks)

    prompt = f"""You are summarising a personal reading log for the past {days} days.

Here are all the entries the user read this period:

{context}

Please generate a structured weekly summary with:
1. A brief overview sentence (what were the main topics this week?)
2. Key themes or concepts that appeared across multiple entries
3. The most interesting or surprising idea encountered
4. A one-sentence "this week in a nutshell" conclusion

Be specific — reference actual content from the entries, not just generic observations.
Format it nicely with clear sections."""

    return call_llm(prompt)


# ── Manual test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(generate_weekly_summary(days=30))  # use 30 days for testing
