# 2026-06-11 — Table-aware extraction (pages 4-27)

Downloaded the HSBC *Pillar 3 Disclosures at 31 December 2025* PDF (122 pp,
`data/pillar3-2025.pdf`) and built `extract_tables.py` to extract pages 4-27
(Highlights, Key Metrics, Own Funds, Leverage, Liquidity) with table structure
preserved. Output: `data/extracted/page-XX.md` (human review) and
`page-XX.json` (structured: caption + columns[header,unit] + rows[ref,label,
values{col→value}]).

## Key decision: word-geometry, NOT a grid/lattice extractor

These regulatory tables have **horizontal rules but no vertical column rules**.
Tested first:
- `pdfplumber.extract_tables` (lines strategy) — merged adjacent cells whenever a
  value was blank, e.g. `15,905 6,989 1,751 392` collapsed into one cell.
- text strategy — split words mid-token (`Available capital ($bn)` → fragments).
- **Camelot would hit the same wall**: `lattice` needs ruling lines (absent);
  `stream` re-implements the same whitespace guessing that scrambles here.

So the table parser is built on pdfplumber's **word coordinates**. Because every
value column is right-aligned, clustering the right edges (`x1`) of numeric
tokens and keeping clusters that recur across many rows isolates the true value
columns cleanly; the far-left row-number gutter is dropped by the large x-gap
that separates it from the data block. Camelot/pdfplumber-tables remain in
`requirements.txt` for tables elsewhere in the document that *are* fully ruled.

## Mechanics that made it robust
- **Columns** from recurring right-edge clusters (freq filter + gutter-gap drop).
- **Values** assigned by nearest column edge (tolerance 22pt).
- **Headers** (e.g. `31 Dec 2025`, `$m`, `Total own funds requirements`)
  rebuilt by *band* assignment — token center within a column's left→right edge
  band — so the right-aligned leading day digit doesn't fall into the prior
  column. Date forward-fill covers headers printed once above paired sub-columns.
- **Units** `$bn / % / $m` captured per-column when aligned, or kept as in-body
  section banners (KM1 style: `Available capital ($bn)`).
- **Wrapped labels** stitched (Market risk: "Position, foreign exchange and
  commodities risks" + "(market risk)").
- **Prose vs table**: tables anchored on `Table N:` captions; a table ends at
  the last row whose numbers align to its columns (contiguous run, small-gap
  tolerant), so trailing **two-column narrative** — whose right column sits in
  the value zone — falls through to the page's `Narrative` section instead of
  polluting the grid.

## Verified (eyeball + arithmetic)
- **KM1 (Table 1, p6)** — 5 quarter-end columns 31 Dec 2025 … 31 Dec 2024.
  CET1 capital (row 1): 132.6 / 127.8 / 129.8 / 125.5 / 124.9 ($bn).
  CET1 ratio (row 5): 14.9 / 14.5 / 14.6 / 14.7 / 14.9 (%).
- **OV1 / "RWAs by risk type" (Table 7, p19)** — 6 cols (RWAs + own-funds × 3
  dates, $m). Credit 675,976 · CCR 42,380 · Market 38,490 · Operational 120,716
  · Total 888,647. Cross-check: risk-type RWAs sum to 888,647 = Total, and
  = KM1 Total RWAs 888.6 $bn. Columns did not scramble.
- QA sweep of all 17 tables: no low-value-row breakage.

## Known limitation
Complex multi-level header tables can have minor header-label gaps — e.g. LI2
(Table 5, p12) renders the first column header as `col0` instead of `Total`, and
occasionally drops the last column's `—` cell. **Values stay aligned**; only the
header label is affected. Revisit if these LI1/LI2 tables become retrieval
targets. KM1 and OV1 (the priority tables) are exact.

## Next
- Decide chunking: one chunk per table (markdown) + per-section narrative chunks.
- Attach metadata (page, table caption, UK code) for citation in the RAG answers.
