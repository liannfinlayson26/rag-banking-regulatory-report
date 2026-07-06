## Prompt 4: Structure-aware chunking (Phase 2)

Build a structure-aware chunking strategy that fixes the table-related breakage
found in the stress test, in a module chunking.py. Work from the verified
extraction in data/extracted/ (do NOT re-parse the PDF).

Critical for an apples-to-apples comparison: embed with the SAME Gemini model used
for the baseline (gemini-embedding-001, 3072-dim) and write to a SEPARATE Chroma
collection "pillar3_structured" — do NOT overwrite "pillar3_baseline". Both stores
must coexist so the before/after is reproducible.

Chunking requirements:
- A table (or coherent section of a large table) stays in ONE chunk with: its table
  number/code (e.g. "Table 7"/"OV1"), caption, column headers INCLUDING reporting
  dates, units row, and row labels. Never split so a number loses its date, unit, or
  row label.
- Where a table pairs an RWA column with a "Total own funds requirements" column per
  date, keep BOTH columns in the same chunk so the 8%-of-RWA relationship is
  reconstructable from one chunk.
- Narrative prose chunks normally.

Chunk metadata (the stress-test failures were about dates/units, so retrieval must
carry them): source_page, table_number/table_code, units present ($m/$bn/%/bps —
list all), reporting_periods present (list of dates), content_type ("table"/"narrative").

Then re-run the SAME 10 stress-test questions through the still-BASELINE (non-agentic)
pipeline against "pillar3_structured", and write docs/chunking-impact.md with a
before/after: which answers improved purely from better chunking, and which still
fail (these justify the agentic layer). Be explicit about units — RWA cells are $m
(Total 888,647 $m = 888.6 $bn); flag any answer right on the figure but wrong on the
unit, since that's a distinct failure class we haven't yet observed.

Do NOT add grading, routing, rewriting, or web search — this phase is still the
baseline pipeline; only chunking and the collection change.