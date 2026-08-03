"""
embeddings/embed.py
────────────────────
Layer 1 — Step 4 & 5: Chunks → embeddings → stored in Chroma.

What is an embedding?
─────────────────────
An embedding is a dense numeric vector (a list of floats) that captures the
*meaning* of a piece of text. The model is trained so that texts with similar
meanings produce vectors that are close together in vector space, and texts
with different meanings produce vectors that are far apart.

Why sentence-transformers/all-MiniLM-L6-v2?
────────────────────────────────────────────
  - Free, runs locally on CPU (no GPU needed)
  - Fast: ~14,000 sentences/second on CPU
  - 384-dimensional output vectors (compact but expressive)
  - 256-token context window — matches our CHUNK_SIZE of 250 words well
  - Very well benchmarked for semantic similarity tasks

Why Chroma?
───────────
  - Free, local, persistent (saves to disk automatically)
  - Simple Python API — no server to run
  - Handles cosine similarity search internally
  - Easy to swap for Pinecone/Weaviate later if you want cloud hosting

Usage
─────
    from embeddings.embed import add_chunks_for_entry, query_similar

    # After chunking an entry:
    add_chunks_for_entry(entry_id=1, chunks=["chunk one text", "chunk two text"])

    # At query time (Layer 2):
    results = query_similar("What is retrieval augmented generation?", top_k=5)
"""

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load once at module level — this avoids reloading the model on every call
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazily load and cache the embedding model."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ── Chroma client ─────────────────────────────────────────────────────────────
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

_client: chromadb.PersistentClient | None = None
_collection = None


def get_collection():
    """Get (or create) the Chroma collection for Daily Brain chunks."""
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if _collection is None:
        _collection = _client.get_or_create_collection(
            name="daily_brain_chunks",
            metadata={"hnsw:space": "cosine"},  # use cosine similarity
        )
    return _collection


# ── Public API ────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into embedding vectors.

    Parameters
    ----------
    texts : list[str]
        The texts to embed.

    Returns
    -------
    list[list[float]]
        One embedding vector per input text.
    """
    model = get_model()
    # encode() returns a numpy array of shape (len(texts), embedding_dim)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def add_chunks_for_entry(entry_id: int, chunks: list[str]) -> None:
    """
    Embed a list of text chunks and store them in Chroma.

    Each chunk is stored with:
      - A unique ID: "entry_{entry_id}_chunk_{i}"
      - The chunk text as document
      - entry_id as metadata (so we can filter by entry later)

    Parameters
    ----------
    entry_id : int
        The ID of the parent entry in the SQLite `entries` table.
    chunks : list[str]
        List of chunk strings (output of chunker.chunk_text).
    """
    if not chunks:
        return

    collection = get_collection()
    embeddings = embed_texts(chunks)

    ids = [f"entry_{entry_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"entry_id": entry_id} for _ in chunks]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"✅ Stored {len(chunks)} chunks for entry {entry_id}")


def query_similar(query: str, top_k: int = 5) -> list[dict]:
    """
    Find the top-k most semantically similar chunks to a query string.

    This is the heart of Layer 2 (RAG) — given a question, return the chunks
    most likely to contain the answer.

    Parameters
    ----------
    query : str
        The question or search string.
    top_k : int
        Number of results to return. Default: 5.

    Returns
    -------
    list[dict]
        Each dict has keys: "chunk_text", "entry_id", "distance", "chunk_id"
        Sorted by relevance (most similar first).
    """
    collection = get_collection()
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist, chunk_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        output.append({
            "chunk_text": doc,
            "entry_id": meta["entry_id"],
            "distance": dist,      # lower = more similar (cosine distance)
            "chunk_id": chunk_id,
        })

    return output


# ── Manual test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Add a test chunk
    test_chunks = [
        "Transformers use self-attention to process sequences in parallel.",
        "RAG combines retrieval with language model generation.",
        "SQLite is a file-based relational database with no separate server.",
    ]
    add_chunks_for_entry(entry_id=9999, chunks=test_chunks)

    # Query it
    results = query_similar("How do transformers handle sequences?", top_k=2)
    for r in results:
        print(f"[dist={r['distance']:.3f}] {r['chunk_text'][:80]}")
