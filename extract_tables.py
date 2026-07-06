#!/usr/bin/env python3
"""Structure-preserving extraction for the HSBC Pillar 3 disclosure (pages 4-27).

These regulatory tables have horizontal rules but no vertical column rules, so
grid-based extractors (pdfplumber `extract_tables`, Camelot lattice/stream)
merge or scramble cells. Instead we rely on the fact that pdfplumber recovers
words *with coordinates* and that every numeric column is right-aligned:

  1. Group words into visual rows by their y position.
  2. Detect data columns by clustering the RIGHT edges (x1) of numeric tokens
     and keeping clusters that recur across many rows (the real value columns),
     dropping the far-left row-number gutter (separated by a large x-gap).
  3. Assign each numeric token to its nearest column; the remaining left-hand
     text is the row label (its leading "1" / "UK-7a" / "14b" becomes `ref`).
  4. Rebuild multi-token / multi-row headers (e.g. "31 Dec 2025", "$m",
     "Total own funds requirements") by bucketing header tokens into the same
     columns, with date forward-fill for headers that span sub-columns.

Output (per page, in document order) goes to data/extracted/:
  * page-XX.md   — captions, prose, and each table rendered as clean markdown
  * page-XX.json — same tables as structured dicts (row label + column -> value)
Prose is kept in separate blocks from tables.
"""
from __future__ import annotations
import json
import re
import statistics as st
from dataclasses import dataclass, asdict, field
from pathlib import Path

import pdfplumber

PDF_PATH = Path("data/pillar3-2025.pdf")
OUT_DIR = Path("data/extracted")
PAGES = range(4, 28)  # 1-based, inclusive: pages 4..27

# --- token classifiers ---------------------------------------------------
NUM_RE = re.compile(r"^\(?-?[\d,]+(?:\.\d+)?\)?%?$|^—$")          # 132.6  1,621.0  (12)  14.9%  —
REF_RE = re.compile(r"^(?:UK-?|EU-?)?\d+[a-z]?$", re.I)            # 1  14b  UK-7a  EU-14d
UNIT_RE = re.compile(r"^\(?(?:\$bn|\$m|%|bps)\)?$", re.I)
SECTION_UNIT_RE = re.compile(r"\((?:\$bn|\$m|%|bps)\)", re.I)  # "...capital ($bn)" banner
MONTH_RE = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b")
CAPTION_RE = re.compile(r"^Table\s+\d+[A-Za-z]?:")
FOOTER_RE = re.compile(r"HSBC Holdings plc", re.I)
Y_TOL = 3.0          # rows within this many points share a baseline
EDGE_GAP = 6.0       # right edges within this -> same column
GUTTER_GAP = 80.0    # x-gap larger than this separates label gutter from data
ASSIGN_TOL = 22.0    # numeric token counts as a cell if within this of a column


@dataclass
class Column:
    x: float                       # representative right-edge x
    header: str = ""               # reconstructed multi-row header
    unit: str = ""                 # $bn / % / $m / bps


@dataclass
class Row:
    ref: str
    label: str
    values: dict                   # column-header -> value string
    is_section: bool = False       # label-only banner row (e.g. "Available capital ($bn)")


@dataclass
class Table:
    caption: str
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)


# --- low-level page parsing ----------------------------------------------
def page_rows(page):
    """Return visual rows: list of (top, [words sorted by x0])."""
    words = page.extract_words(x_tolerance=1.5, keep_blank_chars=False)
    buckets: dict[int, list] = {}
    for w in words:
        buckets.setdefault(round(w["top"] / Y_TOL), []).append(w)
    rows = []
    for _, ws in sorted(buckets.items()):
        ws.sort(key=lambda w: w["x0"])
        rows.append((min(w["top"] for w in ws), ws))
    return rows


def cluster(values, gap):
    values = sorted(values)
    groups = [[values[0]]]
    for v in values[1:]:
        if v - groups[-1][-1] <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups


def detect_columns(block_rows):
    """Detect right-aligned value columns within a block of rows.

    Only rows carrying >=2 numbers are considered, so stray numbers inside
    two-column narrative prose don't spawn phantom columns."""
    data_like = [(t, ws) for t, ws in block_rows
                 if sum(bool(NUM_RE.match(w["text"])) for w in ws) >= 2]
    edges = [w["x1"] for _, ws in data_like for w in ws if NUM_RE.match(w["text"])]
    if len(edges) < 4:
        return []
    groups = cluster(edges, EDGE_GAP)
    n_numeric_rows = len(data_like)
    threshold = max(3, int(0.3 * n_numeric_rows))
    centers = sorted(st.mean(g) for g in groups if len(g) >= threshold)
    if not centers:
        return []
    # Drop the far-left row-number gutter: split at the largest gap > GUTTER_GAP
    # and keep the right-hand (data) block.
    gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
    if gaps:
        biggest, idx = max(gaps)
        if biggest > GUTTER_GAP:
            centers = centers[idx + 1:]
    return [Column(x=c) for c in centers]


def nearest_col(x1, columns):
    best = min(range(len(columns)), key=lambda i: abs(columns[i].x - x1))
    return best if abs(columns[best].x - x1) <= ASSIGN_TOL else None


def band_col(center, columns):
    """Assign a token to the column whose horizontal band contains its center.

    Used for multi-token headers (e.g. "31 Dec 2025"): the leading day digit is
    right-aligned well left of the column's right edge, so a nearest-edge match
    would mis-bucket it. Bands run from one column's right edge to the next."""
    diffs = [columns[i + 1].x - columns[i].x for i in range(len(columns) - 1)]
    spacing = st.median(diffs) if diffs else 50.0
    for i, c in enumerate(columns):
        lo = columns[i - 1].x if i else c.x - spacing
        if lo < center <= c.x + 3:
            return i
    return None


def build_headers(header_rows, columns):
    """Bucket header tokens into columns; concatenate top-to-bottom; fill dates."""
    parts = [[] for _ in columns]      # encounter order preserved (top->bottom)
    units = ["" for _ in columns]
    for _, ws in header_rows:          # header_rows already in vertical order
        for w in ws:
            t = w["text"]
            if t == "At":
                continue               # the "At" banner is not a column label
            i = band_col((w["x0"] + w["x1"]) / 2, columns)
            if i is None:
                continue
            if UNIT_RE.match(t):
                units[i] = t.strip("()")
            else:
                parts[i].append(t)
    headers = [" ".join(toks).strip() for toks in parts]
    # Forward-fill a date across columns that share it (fallback for headers
    # where a date is printed once above paired sub-columns).
    last_date = ""
    for i, h in enumerate(headers):
        m = re.search(r"\d{1,2}\s+\w{3}\s+\d{4}", h)
        if m:
            last_date = m.group(0)
        elif last_date and not re.search(r"\d{4}", h):
            headers[i] = f"{last_date} {h}".strip()
    # Guarantee unique, non-empty column keys so values never collide.
    seen = {}
    for i, h in enumerate(headers):
        h = h or f"col{i}"
        if h in seen:
            seen[h] += 1
            h = f"{h} #{seen[h]}"
        else:
            seen[h] = 1
        columns[i].header, columns[i].unit = h, units[i]
    return columns


# --- table assembly -------------------------------------------------------
def split_blocks(rows):
    """Segment a page into tables (anchored on 'Table N:' captions) and prose.

    Returns (blocks, prose_idx) where each block is (caption, body_rows, columns).
    A table ends at the last row whose numbers actually align to its detected
    columns; rows after that (e.g. two-column narrative whose right column sits
    in the value zone) fall through to prose."""
    captions = [i for i, (_, ws) in enumerate(rows)
                if CAPTION_RE.match(" ".join(w["text"] for w in ws))]
    if not captions:
        return [], list(range(len(rows)))
    blocks, consumed = [], set()
    for n, start in enumerate(captions):
        end = captions[n + 1] if n + 1 < len(captions) else len(rows)
        block = rows[start:end]
        body = block[1:]
        columns = detect_columns(body)
        if not columns:
            consumed.add(start)            # just the caption line; rest is prose
            blocks.append((" ".join(w["text"] for w in block[0][1]), [], []))
            continue
        # Real data rows have >=2 column-aligned numbers and are not date headers
        # (whose day/year digits also align); excluding headers lets the run
        # anchor on the first body row, past the multi-row column header.
        data_idx = [j for j, (_, ws) in enumerate(body)
                    if not MONTH_RE.search(" ".join(w["text"] for w in ws))
                    and sum(1 for w in ws if NUM_RE.match(w["text"])
                            and nearest_col(w["x1"], columns) is not None) >= 2]
        if not data_idx:
            consumed.add(start)
            blocks.append((" ".join(w["text"] for w in block[0][1]), [], []))
            continue
        # Table = contiguous run of aligned rows from the first one, tolerating
        # short gaps (section banners / wrapped labels). A large gap marks the
        # end; isolated aligned rows in trailing two-column prose are excluded.
        MAX_GAP = 4
        last = data_idx[0]
        for j in data_idx[1:]:
            if j - last <= MAX_GAP:
                last = j
            else:
                break
        cap_text = " ".join(w["text"] for w in block[0][1])
        blocks.append((cap_text, body[:last + 1], columns))
        consumed.update(range(start, start + 1 + last + 1))  # caption + table rows
    prose = [i for i in range(len(rows)) if i not in consumed]
    return blocks, prose


def _row_values(ws, columns, left_bound):
    """Split a row's words into (values dict-by-index, leftover label words)."""
    vals, label_toks = {}, []
    for w in ws:
        if NUM_RE.match(w["text"]) and w["x1"] >= left_bound:
            i = nearest_col(w["x1"], columns)
            if i is not None:
                vals[i] = w["text"]
                continue
        label_toks.append(w)
    return vals, label_toks


def parse_table(caption, block_rows, columns):
    if not columns or not block_rows:
        return None
    left_bound = min(c.x for c in columns) - 45

    # Classify each row: a "column-header" row carries a month/date or has >=2
    # tokens aligned under the value columns (sub-headers like RWAs / $m). A
    # "data" row has numeric values and is NOT a date row.
    classified = []
    for top, ws in block_rows:
        text = " ".join(w["text"] for w in ws)
        aligned = sum(1 for w in ws if nearest_col(w["x1"], columns) is not None)
        is_date = bool(MONTH_RE.search(text))
        vals, _ = _row_values(ws, columns, left_bound)
        has_value = bool(vals) and not is_date
        is_colheader = is_date or (aligned >= 2 and not has_value)
        classified.append((top, ws, is_colheader, has_value))

    first_data = next((k for k, c in enumerate(classified) if c[3]), len(classified))
    header_rows = [(t, ws) for k, (t, ws, ch, hv) in enumerate(classified)
                   if ch and k < first_data]
    columns = build_headers(header_rows, columns)
    colnames = [c.header for c in columns]

    body = [(t, ws, hv) for k, (t, ws, ch, hv) in enumerate(classified)
            if not (ch and k < first_data)]

    out_rows = []
    for top, ws, has_value in body:
        vidx, label_toks = _row_values(ws, columns, left_bound)
        if label_toks and REF_RE.match(label_toks[0]["text"]):
            ref = label_toks.pop(0)["text"]
        else:
            ref = ""
        label = " ".join(w["text"] for w in label_toks).strip()
        if FOOTER_RE.search(label):
            continue
        values = {colnames[i]: v for i, v in vidx.items()}
        is_section = bool(label) and not values and bool(SECTION_UNIT_RE.search(label))

        # Merge a wrapped/continuation label: if this row carries values but the
        # previous emitted row was a plain label-only line (not a unit/section
        # banner), the label wrapped across lines -> stitch them together.
        if values and out_rows:
            prev = out_rows[-1]
            if not prev.values and prev.label and not prev.is_section:
                label = (prev.label + " " + label).strip()
                ref = ref or prev.ref
                out_rows.pop()

        if (not label and not values) or label in {"At", "At At", "At At At"}:
            continue
        # Drop prose/footnotes that leaked in as value-less rows (a numeric token
        # inside a sentence can extend the block). Real labels and section
        # banners are short and don't read as sentences.
        if not values and not is_section and (label.endswith(".") or len(label.split()) > 9):
            continue
        out_rows.append(Row(ref=ref, label=label, values=values, is_section=is_section))
    return Table(caption=caption, columns=columns, rows=out_rows)


# --- rendering ------------------------------------------------------------
def table_to_markdown(t: Table) -> str:
    cols = t.columns
    head = ["", ""] + [c.header for c in cols]
    units = ["", ""] + [f"({c.unit})" if c.unit else "" for c in cols]
    lines = []
    lines.append("| " + " | ".join(["Ref", "Item"] + [h for h in head[2:]]) + " |")
    lines.append("| " + " | ".join(["---"] * (len(cols) + 2)) + " |")
    if any(u.strip("()") for u in units[2:]):
        lines.append("| " + " | ".join(units) + " |")
    for r in t.rows:
        cells = [r.ref, r.label] + [r.values.get(c.header, "") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def prose_text(rows, prose_idx):
    out = []
    for i in prose_idx:
        _, ws = rows[i]
        line = " ".join(w["text"] for w in ws).strip()
        if not line or FOOTER_RE.search(line) or CAPTION_RE.match(line):
            continue
        if re.fullmatch(r"\d{1,3}", line):  # bare page number
            continue
        out.append(line)
    return "\n".join(out)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(PDF_PATH) as pdf:
        for pno in PAGES:
            rows = page_rows(pdf.pages[pno - 1])
            blocks, prose_idx = split_blocks(rows)
            tables = []
            for caption, block_rows, columns in blocks:
                t = parse_table(caption, block_rows, columns)
                if t and t.rows:
                    tables.append(t)
            prose = prose_text(rows, prose_idx)

            md = [f"# Page {pno}\n"]
            if prose:
                md.append("## Narrative\n\n" + prose + "\n")
            for t in tables:
                md.append("## " + t.caption + "\n\n" + table_to_markdown(t) + "\n")
            (OUT_DIR / f"page-{pno:02d}.md").write_text("\n".join(md))

            payload = {
                "page": pno,
                "prose": prose,
                "tables": [
                    {
                        "caption": t.caption,
                        "columns": [asdict(c) for c in t.columns],
                        "rows": [asdict(r) for r in t.rows],
                    }
                    for t in tables
                ],
            }
            (OUT_DIR / f"page-{pno:02d}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Extracted pages {PAGES.start}-{PAGES.stop - 1} -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
