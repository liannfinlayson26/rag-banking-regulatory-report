# 2026-06-11 — Baseline (non-agentic) RAG

Built `baseline_rag.ipynb`: the deliberately-naive control pipeline we will measure
the structure-aware chunking against. Executed end-to-end; outputs are saved in the
notebook and the store is persisted to `chroma_db/`.

## Pipeline
- **Source:** `data/extracted/page-*.md` (pages 4–27), one `Document` per page,
  `source_page` metadata only.
- **Chunking (naive on purpose):** `RecursiveCharacterTextSplitter`, size 800 /
  overlap 100, generic prose separators — structure-blind. 24 pages → **210 chunks**.
- **Embeddings:** Gemini `models/gemini-embedding-001` (3072-dim). `text-embedding-004`
  returned 404 for this key; `gemini-embedding-001` works.
- **Store:** Chroma persisted to `chroma_db/`, collection `pillar3_baseline`,
  `hnsw:space=cosine`. **These are the reference collection settings** the later
  structure-aware store must reuse.
- **Retrieve:** top-k = 4.
- **Generate:** `gpt-4o-mini`, temperature 0, answering only from retrieved context;
  prompt asks for unit + reporting date.

## Deliberately absent
No grading, routing, query rewriting, or web search. One-shot, no retrieval
verification, no recovery — by design.

## Operational note
Gemini's **free** embedding tier is ~100 requests/minute; embedding all 210 chunks at
once threw 429 RESOURCE_EXHAUSTED. Fixed in-notebook with throttled batches (50/batch,
61s pause) + 429-retry. Full build ≈ 4½ min.

## Baseline behaviour already visible (smoke test)
- *"CET1 capital ratio at 31 Dec 2025?"* → **"14.9%"** ✓ — answerable from Highlights/
  KM1 prose.
- *"RWAs for credit risk (excl. CCR)?"* → **"does not contain enough information"** ✗ —
  it retrieved page 19 (OV1/Table 7) but the naive split severed the `675,976` cell
  from its row label and the date/unit header, so the figure was unusable. This is the
  table-breakage failure mode the baseline exists to expose.

## Next
- Structure-aware `chunking.py` (keep table + caption + headers/dates + units + row
  labels together; keep OV1 RWA & own-funds columns in one chunk).
- Define the 10 stress-test questions; run the SAME baseline pipeline against both
  stores; write `docs/chunking-impact.md` (before/after, flag figure-right/unit-wrong
  as a distinct class).

## Dev deps added
`langchain-text-splitters==1.1.2` (core), `ipykernel==7.3.0`, `nbconvert==7.17.1`
(to execute notebooks). `requirements.txt` + `requirements.lock.txt` updated.
