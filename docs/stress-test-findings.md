# Stress-test findings — naive baseline (2026-06-11)

10 questions run through the **existing** baseline pipeline against the
**already-built** Chroma collection `pillar3_baseline` (210 naive chunks, reused
as-is — no re-chunk/re-embed/rebuild). Gemini embeddings · top-k = 4 ·
`gpt-4o-mini` @ temp 0 · grounded on retrieved context only. No grading, routing,
rewriting, or web search. The model saw **only the bare question**; gold was used
solely for post-hoc scoring.

> **All page references below are ACTUAL PDF pages** (what a PDF viewer's go-to-page
> opens). The canonical per-question source list (PDF page + the document's own
> printed page) is in [`eval_set.json`](../eval_set.json).

## Gold verification (against `data/extracted/`, PDF pages 4–27)
All 10 gold answers were verified present in the extracted store — **no factual
mismatches**. Caveats flagged rather than overwritten:
- **Page numbering (corrected):** citations now use **actual PDF pages**. The
  document's own *printed* footer number = **PDF page − 1**, uniformly across the
  extracted range (verified: PDF p19 shows printed "18"; PDF p5 shows printed "4").
  The gold's original printed-page refs were shifted accordingly: "Highlights p4" →
  **PDF p5**, "Table 1 p5" → **PDF p6**, "Article 92 p3" → **PDF p4**. PDF p4
  literally reads *"minimum total capital charge set at 8% of risk-weighted assets
  ('RWAs') by Article 92(1) of CRR II"* — so Q8 **is** retrievable.
- **Q9 currency is split from the figure:** `888,647` + `($m)` are on **PDF p19**
  (Table 7/OV1), but "US dollars" only appears on **PDF p20/p27**. Grounding the
  currency needs a *different* chunk than the figure. (A front-matter currency
  definition likely sits on PDF p2, outside the extracted 4–27 range.)
- **Computed golds** (Q4 sum `80,870`, Q10 delta `14,244`) are not in the text by
  design — they require arithmetic.

## Scoreboard — 3 / 10 pass

| Q | verdict | retrieved right page? | failure type |
|---|---|---|---|
| 1 CET1 ratio | ✗ refused | **yes** (p6 KM1, p5 Highlights) | fragmentation → refusal |
| 2 CET1 capital $bn | ✓ **pass** | yes | — |
| 3 leverage ratio + direction | ✓ **pass** | yes | — (temporal trap avoided) |
| 4 market+CCR RWA | ✗ refused | yes (p19 OV1) | fragmentation → cells not in top-k |
| 5 transitional vs end-point | ✗ refused | yes (p6 preamble) | fragmentation → caveat not in top-k |
| 6 LCR / NSFR / HQLA | ✗ **wrong** | yes (p5, p6) | row-label loss → conflation |
| 7 Hang Seng bps + when | ✓ **pass** | yes | — (110/120/10 conflation avoided) |
| 8 total capital charge + reg | ✗ **wrong** | yes (p7 has 8%/Art.92) | concept + cross-ref loss |
| 9 total RWA + currency | ✗ **wrong** | yes (p19) | wrong cell read as total |
| 10 op-risk RWA + Δ vs 2024 | ✗ **wrong** | yes (p19) | date-column stripped → wrong year |

Two distinct buckets among the 7 failures: **3 honest refusals** (no fabrication)
and **4 confidently wrong** answers.

## Evidence — actual baseline answers

**✓ Q2** — *"HSBC's CET1 capital at 31 Dec 2025 was $132.6bn."* — correct, right unit.

**✓ Q3** — *"…5.3% at 31 December 2025, down from 5.6% at 31 December 2024."* —
correct; did **not** fall for the "down from" temporal trap.

**✓ Q7** — *"…net CET1 capital impact … was 110 bps upon taking effect in January
2026."* — correct core fact; avoided the 110/120/10 bps conflation (omits the
day-one detail, but the asked figure + timing are right).

**✗ Q1** (refused) — gold 14.9%. The KM1 page *and* the Highlights page were both in
the top-4, yet the model refused. (Note: the earlier smoke test, with the phrasing
*"What **was** the CET1 capital ratio…"*, answered "14.9%". Same fact, different
phrasing → different chunks retrieved → different outcome — baseline fragility.)

**✗ Q4** (refused) — gold 38,490 + 42,380 = 80,870 $m. Page 19 (OV1) was retrieved,
but the specific 800-char chunks did **not** contain the market-risk / CCR cells
(the table was shattered into many fragments), so there was nothing to aggregate.

**✗ Q5** (refused) — gold "identical on both bases". Page 6 was retrieved, but the
preamble sentence explaining the transitional/end-point equivalence was split into a
different chunk than the one retrieved.

**✗ Q6** (wrong) — *"LCR … was 143%, and the NSFR was also 143%. The HQLA … $702bn."*
NSFR (143%) and HQLA ($702bn) are right, but **LCR is 137%, not 143%** — the chunk
carried `137` and `143` without their `LCR (%)` / `NSFR (%)` row labels, so the model
mis-assigned. Classic **row-label-loss conflation**, and it silently dropped a part.

**✗ Q8** (wrong) — *"…total capital charge of 20.5% of RWAs under … Basel III…"*
Gold is **8%, Article 92(1) CRR II**. `20.5%` is HSBC's total capital *ratio*
(KM1 row 7) — the model grabbed a nearby ratio instead of the 8% minimum *charge*,
and cited "Basel III" generically. Concept confusion **plus** the lost cross-reference.

**✗ Q9** (wrong) — *"…total RWAs were $22,308m."* Gold is **$888,647m**. `22,308` is
the market-risk *internal-models* sub-row of OV1 — a stray cell, read as the total
because its row label was gone. Also omits the currency. A severe **wrong-cell**
failure; the $m/$bn unit risk never even arose because the figure itself was wrong.

**✗ Q10** (wrong) — *"$120,716m … increased by $11,685m from $109,031m in 2024."*
Current value `120,716` is right, but **`109,031` is the 30 Sep 2025 column, not
2024** (2024 = `106,472`). With the date headers stripped, the model picked the wrong
column → wrong delta (`11,685` vs true `14,244`). The canonical **date-stripping**
failure.

## The key isolation: chunking, not retrieval/embedding
In **every** failure the correct page was in the top-4. Embeddings and retrieval are
not the bottleneck — **naive chunking is**. The damage takes three forms:
1. **Date stripping** → number mapped to the wrong reporting period (Q10).
2. **Row-label loss** → numbers conflated across rows (Q6), or a stray cell read as
   the headline figure (Q9, Q8).
3. **Fragmentation** → a table/preamble shattered into many low-signal 800-char
   pieces, so the needed cell/sentence never reaches the top-4 (Q1, Q4, Q5).

The unit failure class the brief asked us to watch for (right figure, wrong unit —
e.g. `888,647 $bn`) did **not** surface, because a more basic failure pre-empted it
(Q9 returned the wrong cell entirely). The risk remains real: the structure-aware
chunker must keep each cell with its `$m`/`$bn`/`%` unit.

## What this justifies next
- **Fixable by structure-aware chunking** (keep caption + headers + dates + units +
  row labels together, and OV1's RWA/own-funds columns in one chunk): Q1, Q5, Q6,
  Q9, Q10, and the *inputs* for Q4 — these failed only because structure was lost on
  a page that was correctly retrieved.
- **Still needs the agentic layer** (reasoning/verification beyond chunking): the
  arithmetic in Q4 (sum) and Q10 (delta); Q8's intent disambiguation ("minimum
  charge" vs "ratio"); and Q9's currency, which lives on a different page and needs
  multi-chunk synthesis or routing. These are the cases that justify grading,
  self-check, and routing later — not just better chunks.

This is the measured "before". The structure-aware chunker is expected to flip the
first group to pass; the second group is the evidence base for the agentic phase.
