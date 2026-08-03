"""
db/schema.py
────────────
Defines and initialises the SQLite database for Daily Brain.

Tables
──────
  entries  — one row per thing you feed the system each day
  chunks   — one row per text chunk derived from an entry (added in Layer 1)

Run directly to (re)create the schema:
    python3 db/schema.py
"""

import sqlite3
from pathlib import Path

# Resolve DB path relative to this file so it always works regardless of CWD
DB_PATH = Path(__file__).parent / "daily_brain.sqlite"


def get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # lets you access columns by name: row["date_added"]
    conn.execute("PRAGMA foreign_keys = ON")  # enforce FK constraints
    return conn


def init_db() -> None:
    """Create tables if they don't already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # ── entries ───────────────────────────────────────────────────────────────
    # One row per thing you feed the system each day.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            date_added   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source_type  TEXT NOT NULL CHECK(source_type IN ('article','paper','note','screenshot')),
            raw_text     TEXT NOT NULL,
            cleaned_text TEXT,
            source_url   TEXT
        )
    """)

    # ── chunks ────────────────────────────────────────────────────────────────
    # One row per text chunk derived from an entry.
    # embedding is stored as a binary blob (serialised numpy array).
    # Added in Layer 1 — this table can be empty until you build chunker.py.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id   INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
            chunk_text TEXT NOT NULL,
            embedding  BLOB                  -- null until Layer 1 embedding step
        )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Database initialised at: {DB_PATH}")


if __name__ == "__main__":
    init_db()
