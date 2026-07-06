# 2026-06-13 — Structure-aware chunking

Built `chunking.py` (works from `data/extracted/*.json`, does **not** re-parse the
PDF) and `compare_chunking.py`. Full before/after analysis: `chunking-impact.md`.

## Design decisions
- **Whole-table chunks.** Each table is one self-contained chunk: caption + table
  number/code + column headers *including reporting dates* + units row + every row
  label, rendered as Markdown. Largest table (KM1) is ~4.5k chars, under the
  embedding limit, so no table needed splitting; a row-group splitter (header block
  repeated, section banner carried) is in place as a safety for tables > 6000 chars.
- **Paired columns stay together.** OV1's RWA + "Total own funds requirements" per
  date are in the same chunk (we never split columns, only — if forced — rows), so
  the 8%-of-RWA relationship is reconstructable from one chunk.
- **Narrative** splits normally (recursive, 1100/150).
- **Metadata (Chroma scalars; lists comma-joined):** `source_page`, `content_type`
  (table/narrative), `table_number`, `table_code`, `units`, `reporting_periods`.

## Store
- Same embedding model as baseline: `models/gemini-embedding-001` (3072-dim), cosine.
- Separate collection **`pillar3_structured`** in the same `chroma_db/`; only that
  collection is reset on rebuild — `pillar3_baseline` (210 vectors) is left intact
  and verified present after the build. 24 pages → **109 chunks** (17 table + 92
  narrative) vs 210 naive.

## Result (same baseline pipeline, both stores)
**3/10 → 6/10.** Clean chunking flips: Q1 (CET1 ratio), Q6 (LCR/NSFR/HQLA — no more
row-label conflation), Q10 (op-risk delta with correct dated columns + `$m`). Q9's
figure+unit fixed (`$888,647m`) but currency still missing. The "right figure / wrong
unit" class still did not appear — structured chunking binds `$m` to every cell, so
it's structurally mitigated.

Remaining failures are no longer chunking (see `chunking-impact.md`): Q4 (data co-
located, needs arithmetic), Q5 (caveat retrieved, needs synthesis), Q8 (right chunk
not ranked top-4 → routing/hybrid), Q9 (currency on another page → multi-hop). These
are the evidence base for the agentic layer.

## Still baseline
No grading, routing, rewriting, or web search added — only chunking and the
collection changed.
