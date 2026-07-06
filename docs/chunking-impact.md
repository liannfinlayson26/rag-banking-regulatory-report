# Chunking impact — naive vs structure-aware (2026-06-13)

Same 10 stress-test questions, **same baseline pipeline** (top-k = 4, `gpt-4o-mini`
@ temp 0, identical grounding prompt, no grading/routing/rewriting/web search), run
against two coexisting Chroma collections:

| | chunks | strategy |
|---|---|---|
| `pillar3_baseline` | 210 | naive fixed-size (800/100), structure-blind |
| `pillar3_structured` | 109 (17 table + 92 narrative) | tables kept whole with caption + dated headers + units + row labels; prose split normally |

Both embed with the **same** model (`models/gemini-embedding-001`, 3072-dim, cosine)
and live in the same `chroma_db/`, so the before/after is reproducible:
`python chunking.py` (builds only the structured collection) then
`python compare_chunking.py` → `data/chunking_comparison.json`.

## Headline: 3 / 10 → 6 / 10

Three clean flips, all **fail → pass**, purely from better chunking: **Q1, Q6, Q10**.
No regressions. Q9 improved materially (figure + unit now correct) but still misses
one part. The remaining four failures now have a *different* diagnosis — they are no
longer about chunking.

| Q | baseline | structured | what changed |
|---|---|---|---|
| 1 CET1 ratio | ✗ refused | ✓ **14.9%** | chunking |
| 2 CET1 capital $bn | ✓ | ✓ | — |
| 3 leverage + direction | ✓ | ✓ | — |
| 4 market+CCR RWA | ✗ refused | ✗ refused | data now present, won't aggregate |
| 5 transitional/end-point | ✗ refused | ✗ refused | caveat now retrieved, not synthesised |
| 6 LCR/NSFR/HQLA | ✗ wrong (LCR 143) | ✓ **137/143/702** | chunking |
| 7 Hang Seng bps | ✓ | ✓ | — |
| 8 capital charge + reg | ✗ wrong (20.5%) | ✗ wrong (15.7%) | right chunk not retrieved |
| 9 total RWA + currency | ✗ wrong ($22,308m) | ✗ **figure+unit fixed**, currency missing | partial |
| 10 op-risk RWA + Δ | ✗ wrong delta | ✓ **120,716 vs 106,472 = +14,244 $m** | chunking |

## Improved purely from chunking (a structure fix, not reasoning)

**Q1 — CET1 ratio.** Baseline *refused* (the 14.9% was retrieved but stranded from
its label/date). Structured: *"…CET1 capital ratio as at 31 December 2025 is 14.9%."*
The whole KM1 table chunk (with the `Capital ratios (%)` banner and the dated column)
plus the Highlights narrative chunk made the fact usable.

**Q6 — LCR / NSFR / HQLA.** Baseline conflated: *"LCR … was 143%, … NSFR … also 143%"*
(it lost the `LCR (%)` / `NSFR (%)` row labels). Structured:
*"LCR … was 137% and the NSFR was 143%. The HQLA amount was $702bn."* Row labels
preserved in the KM1 chunk → no conflation, all three parts present.

**Q10 — operational-risk RWA and change.** Baseline returned the right current figure
but a wrong/ill-scoped delta (*"increased by $11.6bn"* — actually the document's own
narrative figure, not the table year-over-year). Structured:
*"…120,716 ($m) as of 31 Dec 2025. It increased by $14,244 ($m) from 106,472 ($m) in
31 Dec 2024."* With the **dated** headers (`31 Dec 2025` vs `31 Dec 2024`) intact, the
model picked the correct comparison column (not the 30 Sep `109,031` it would
otherwise grab) and computed the delta correctly, with `$m` on every figure.

**Q9 — figure + unit fixed (still fails on completeness).** Baseline returned a stray
OV1 sub-cell as the total: *"total RWAs were $22,308m."* Structured returns the right
cell with the right unit: *"…total RWA is $888,647m as of 31 December 2025."* The
figure and `$m` unit are now correct; it only misses the **currency** (USD), which
lives on a different page (p20/p27) and was not retrieved with the OV1 chunk.

## Still failing — these justify the agentic layer (not more chunking)

The key diagnostic is `retrieved_has_fact` — whether the needed fact was actually in
the retrieved context:

- **Q4 (refused) — needs arithmetic.** `retrieved_has_fact = True`: the OV1 chunk
  brought market-risk `38,490` and CCR `42,380` together in one chunk. The pipeline
  still refused because the *combined* `80,870` is not literally in the text and the
  one-shot grounded prompt won't add two cells. Chunking did its job; the gap is a
  **compute/reasoning step**.
- **Q5 (refused) — needs synthesis.** `retrieved_has_fact = True`: the
  transitional/end-point preamble is now retrievable, but the baseline didn't
  recognise that *"figures are the same on both bases"* **is** the answer. A
  qualitative-synthesis gap, not a structure gap → reasoning / self-check.
- **Q8 (wrong) — needs routing / better retrieval.** `retrieved_has_fact = False`:
  the *"8% … Article 92(1) of CRR II"* sentence is its own (page-4) chunk but did not
  rank top-4; the query pulled capital-requirement table chunks instead, and the model
  answered with the nearby `15.7%` overall requirement. The right chunk exists but
  isn't retrieved → **query routing / hybrid (keyword) retrieval / grade-and-re-retrieve**.
- **Q9 (incomplete) — needs cross-page synthesis.** Figure + unit fixed (above); the
  missing currency sits on a different page → **multi-chunk / multi-hop** retrieval.

## Units — explicit, as requested

RWA cells are **`$m`**: OV1 Total `888,647 $m` = **`888.6 $bn`**. The structure-aware
chunker binds the unit to every cell (each column header carries the date *and* the
unit, plus a units row), so a figure can't be separated from its unit within a chunk.

**The "right figure, wrong unit" failure class still did not appear in either run.**
On the contrary, structured chunking now reports `$m` correctly where it matters:
Q9 → `$888,647m`, Q10 → `120,716 ($m)` / `106,472 ($m)` / `+14,244 ($m)`. The risk the
brief flagged is real but is now **structurally mitigated** — units travel with the
cells — so it remains unobserved rather than latent-and-waiting.

## Conclusion

Structure-aware chunking fixed exactly the failure mode it targeted: numbers losing
their date, unit, or row label (Q1, Q6, Q10), and it corrected the wrong-cell / wrong-
unit hazard on Q9. The four residual failures are no longer chunking problems —
`retrieved_has_fact` is now `True` for Q4 and Q5 (data present, not reasoned over),
and the Q8/Q9 gaps are retrieval-ranking and cross-page-synthesis. Those four are the
concrete evidence base for the **agentic layer**: a compute/reasoning step (Q4, Q10-
style deltas), answer synthesis/self-check (Q5), and query routing / grade-and-re-
retrieve / multi-hop (Q8, Q9) — none of which is "just better chunks".
