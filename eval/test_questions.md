# RAG Evaluation Log
# Daily Brain — Layer 2 Evaluation

This log tracks the quality of Daily Brain's RAG system over time.
Fill it in once you've built Layer 2 and have at least 30+ entries.

---

## How to evaluate

For each test question:
1. Ask Daily Brain the question
2. Check: did it retrieve the **right** chunks?
3. Check: does the LLM answer **accurately reflect** those chunks?
4. Check: for "unknown" questions, does it correctly say "I don't know"?
5. Score each dimension 1-3 and note observations

**Scoring:**
- 3 = Correct / grounded / appropriate
- 2 = Partially correct / minor issues
- 1 = Wrong / hallucinated / retrieved wrong chunks

---

## Test Cases

| # | Question | In knowledge base? | Retrieval score (1-3) | Generation score (1-3) | Grounding score (1-3) | Notes |
|---|----------|-------------------|----------------------|------------------------|----------------------|-------|
| 1 | | Yes | | | | |
| 2 | | Yes | | | | |
| 3 | | Yes | | | | |
| 4 | | Yes | | | | |
| 5 | | Yes | | | | |
| 6 | | Yes | | | | |
| 7 | | Yes | | | | |
| 8 | | Yes | | | | |
| 9 | | Yes | | | | |
| 10 | | Yes | | | | |
| 11 | | **No** | — | — | | Did it correctly say "I don't know"? |
| 12 | | **No** | — | — | | Did it correctly say "I don't know"? |
| 13 | | **No** | — | — | | Did it correctly say "I don't know"? |
| 14 | | Yes | | | | |
| 15 | | Yes | | | | |
| 16 | | Yes | | | | |
| 17 | | Yes | | | | |
| 18 | | Yes | | | | |
| 19 | | **No** | — | — | | |
| 20 | | **No** | — | — | | |

---

## Failure Analysis

Document cases where retrieval failed and **why**:

### Case #__ — [date]
- **Question:** ...
- **What was retrieved:** ...
- **What should have been retrieved:** ...
- **Why it failed:** (e.g., chunk boundary split the relevant sentence, wrong chunk_size, embedding model limitation)
- **Fix tried:** ...

---

## Hallucination Log

Cases where the LLM added information not in the retrieved context:

| Date | Question | What the LLM added | Was it correct? | Prompt fix applied? |
|------|----------|---------------------|-----------------|---------------------|
| | | | | |

---

## Tuning Log

| Date | Parameter changed | Old value | New value | Effect |
|------|-------------------|-----------|-----------|--------|
| | chunk_size | 250 | | |
| | overlap | 30 | | |
| | top_k | 5 | | |
| | LLM model | | | |

---

*This evaluation log is what makes your RAG project stand out — most fresher projects have zero evaluation, just a demo that "looks like it works."*
