"""
agents/multistep_qa.py
──────────────────────
Layer 3 — Agent 3: Multi-turn conversational Q&A grounded in your notes.

Unlike single-shot RAG (one question → one answer), this agent maintains a
conversation history. Each new question retrieves fresh context AND sees the
prior conversation turns — so you can ask follow-up questions naturally.

Agentic pattern
───────────────
  Turn 1:  embed(q1) → retrieve chunks → LLM(system + context + q1) → a1
  Turn 2:  embed(q2) → retrieve NEW chunks → LLM(system + context + history + q2) → a2
  Turn N:  ...

Usage
─────
    from agents.multistep_qa import MultiStepQA

    agent = MultiStepQA()
    print(agent.ask("What is an articulation point?"))
    print(agent.ask("Can you explain the DFS approach for finding them?"))
    agent.reset()  # clear history
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.embed import query_similar
from retrieval.rag import call_llm


class MultiStepQA:
    """
    Stateful multi-turn Q&A agent grounded in the Daily Brain knowledge base.
    """

    def __init__(self, top_k: int = 5, max_history_turns: int = 6):
        self.top_k = top_k
        self.max_history_turns = max_history_turns
        self.history: list[dict] = []

    def ask(self, question: str) -> dict:
        """
        Ask a question. Returns dict with 'answer', 'sources', 'chunks'.
        """
        # Step 1: retrieve fresh context
        chunks = query_similar(question, top_k=self.top_k)

        if chunks:
            context_parts = [
                f"[Source {i+1} — Entry #{c['entry_id']}]\n{c['chunk_text']}"
                for i, c in enumerate(chunks)
            ]
            context_block = "\n\n---\n\n".join(context_parts)
        else:
            context_block = "(No relevant entries found in knowledge base.)"

        source_ids = list({c["entry_id"] for c in chunks}) if chunks else []

        # Step 2: build conversation history string (last N turns)
        recent = self.history[-(self.max_history_turns * 2):]
        history_lines = []
        for turn in recent:
            label = "User" if turn["role"] == "user" else "Assistant"
            history_lines.append(f"{label}: {turn['content']}")
        history_text = "\n".join(history_lines)

        # Step 3: build full prompt
        conv_section = f"\nCONVERSATION SO FAR:\n{history_text}\n" if history_text else ""

        prompt = (
            "You are a personal knowledge assistant in a multi-turn conversation.\n"
            "Ground your answers ONLY in the CONTEXT and prior conversation below.\n"
            "If something is not in the context, say so honestly.\n"
            "You MAY reference what was said earlier in the conversation.\n\n"
            f"CONTEXT (retrieved from user's knowledge base):\n{context_block}\n"
            f"{conv_section}\n"
            f"User: {question}\nAssistant:"
        )

        # Step 4: call LLM
        answer = call_llm(prompt)

        # Step 5: record this turn in history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": source_ids,
            "chunks": chunks,
        }

    def reset(self) -> None:
        """Clear conversation history to start a new session."""
        self.history = []

    @property
    def turn_count(self) -> int:
        """Number of completed Q&A turns in this session."""
        return len(self.history) // 2


# ── Manual test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = MultiStepQA()
    r1 = agent.ask("What is an articulation point?")
    print("Turn 1:", r1["answer"])
    r2 = agent.ask("What is the time complexity of finding them?")
    print("Turn 2:", r2["answer"])
