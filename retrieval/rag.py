"""
retrieval/rag.py
────────────────
Layer 2: Query → Retrieve → Generate

The full RAG pipeline:
  1. Embed the user's question
  2. Find the top-k most similar chunks in Chroma (using embed.query_similar)
  3. Build a prompt that gives the LLM those chunks as context
  4. Call the LLM and return the answer + citations

Grounding instruction (critical)
──────────────────────────────────
The prompt explicitly tells the LLM:
  "Answer ONLY using the provided context. If the answer is not in the
   context, say 'I don't have information about this in my notes.'"

This prevents the LLM from answering from its general training knowledge
instead of your specific data. Without this, the system feels like RAG
but actually isn't — the LLM just ignores your context and answers anyway.

LLM configuration
──────────────────
Set the LLM_PROVIDER env variable (in .env file):
  LLM_PROVIDER=ollama        → uses local Ollama (recommended, free, private)
  LLM_PROVIDER=gemini        → uses Google Gemini API (requires GEMINI_API_KEY)
  LLM_PROVIDER=openai        → uses OpenAI API (requires OPENAI_API_KEY)

Usage
─────
    from retrieval.rag import answer_question

    result = answer_question("What is retrieval augmented generation?")
    print(result["answer"])
    print(result["sources"])   # list of entry_ids used
"""

import os
from dotenv import load_dotenv
from embeddings.embed import query_similar

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

TOP_K = 5  # how many chunks to retrieve


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Build the RAG prompt: context + question + grounding instruction.

    Parameters
    ----------
    question : str
        The user's question.
    context_chunks : list[dict]
        Output of query_similar() — each has "chunk_text" and "entry_id".

    Returns
    -------
    str
        The full prompt to send to the LLM.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        context_parts.append(
            f"[Source {i+1} — Entry #{chunk['entry_id']}]\n{chunk['chunk_text']}"
        )
    context_text = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a personal knowledge assistant. You have access to the user's reading notes below.

CONTEXT (from the user's notes):
{context_text}

QUESTION: {question}

INSTRUCTIONS:
- Answer ONLY using the information in the CONTEXT above.
- If the answer cannot be found in the context, respond with exactly:
  "I don't have information about this in my notes."
- Do not use your general training knowledge to supplement the answer.
- Cite which Source number(s) your answer draws from (e.g. "According to Source 2...").
- Be concise and direct.

ANSWER:"""
    return prompt


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    """
    Send the prompt to the configured LLM and return the response text.

    Supports:
      - Ollama (local, recommended)
      - Google Gemini API
      - OpenAI API
    """
    if LLM_PROVIDER == "ollama":
        return _call_ollama(prompt)
    elif LLM_PROVIDER == "gemini":
        return _call_gemini(prompt)
    elif LLM_PROVIDER == "openai":
        return _call_openai(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}. Set in .env file.")


def _call_ollama(prompt: str) -> str:
    """Call a local Ollama model. Install Ollama from https://ollama.ai"""
    try:
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return f"[Ollama error: {e}]\nMake sure Ollama is running: `ollama serve`"


def _call_gemini(prompt: str) -> str:
    """Call Google Gemini API."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini error: {e}]"


def _call_openai(prompt: str) -> str:
    """Call OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI error: {e}]"


# ── Main RAG pipeline ─────────────────────────────────────────────────────────

def answer_question(question: str, top_k: int = TOP_K) -> dict:
    """
    Full RAG pipeline: question → answer with citations.

    Parameters
    ----------
    question : str
        Plain-English question to answer from your knowledge base.
    top_k : int
        How many chunks to retrieve. Default: 5.

    Returns
    -------
    dict with keys:
        "answer"   : str  — the LLM's answer
        "sources"  : list[int]  — entry IDs used
        "chunks"   : list[dict] — the raw retrieved chunks (for debugging)
    """
    # Step 1 + 2: embed query and retrieve similar chunks
    chunks = query_similar(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "No entries in your knowledge base yet. Add some entries first!",
            "sources": [],
            "chunks": [],
        }

    # Step 3: build grounded prompt
    prompt = build_prompt(question, chunks)

    # Step 4: call LLM
    answer = call_llm(prompt)

    # Step 5: collect source entry IDs for citation
    source_entry_ids = list({c["entry_id"] for c in chunks})

    return {
        "answer": answer,
        "sources": source_entry_ids,
        "chunks": chunks,
    }


# ── Manual test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = answer_question("What is attention in transformers?")
    print("ANSWER:", result["answer"])
    print("SOURCES (entry IDs):", result["sources"])
