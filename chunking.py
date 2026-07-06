#!/usr/bin/env python3
"""Structure-aware chunking for the HSBC Pillar 3 RAG store.

Fixes the table breakage the naive baseline exposed (numbers stranded from their
reporting date / unit / row label). Works ONLY from the verified extraction in
`data/extracted/*.json` — it does not re-parse the PDF.

Design
------
* **Tables** become self-contained chunks. Each carries: the table number/code,
  caption, the column headers *including the reporting dates*, the units row, and
  every row label — rendered as Markdown so a row label and a date column map
  unambiguously to one value. Paired columns (e.g. OV1's RWA + "Total own funds
  requirements" per date) stay in the same chunk, so the 8%-of-RWA relationship is
  reconstructable from a single chunk. A table is kept whole; only a table larger
  than ``MAX_TABLE_CHARS`` is split into row groups, with the full header block
  repeated on each group (and the current section banner carried over).
* **Narrative prose** is chunked normally with a recursive splitter.

Embeddings use the SAME model as the baseline (``models/gemini-embedding-001``,
3072-dim) and are written to a SEPARATE Chroma collection ``pillar3_structured``
in the same ``chroma_db/`` dir. The baseline collection ``pillar3_baseline`` is
left untouched, so the before/after comparison is reproducible.

Run:  python chunking.py        # (re)builds only the structured collection
"""
from __future__ import annotations
import glob
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

EXTRACTED_DIR = "data/extracted"
PERSIST_DIR = "chroma_db"
COLLECTION = "pillar3_structured"          # SEPARATE from "pillar3_baseline"
BASELINE_COLLECTION = "pillar3_baseline"   # must remain intact
DISTANCE = "cosine"                        # same as baseline
EMBED_MODEL = "models/gemini-embedding-001"  # same as baseline (3072-dim)
NARRATIVE_SIZE, NARRATIVE_OVERLAP = 1100, 150
MAX_TABLE_CHARS = 6000                     # split a table only if it exceeds this

DATE_RE = re.compile(r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
                     r"(?:\s+\d{4})\b|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|"
                     r"August|September|October|November|December)\s+\d{4}\b")
UNIT_TOKENS = [("$m", r"\$m\b"), ("$bn", r"\$bn\b"), ("%", r"%"), ("bps", r"\bbps\b")]


# --- metadata helpers -----------------------------------------------------
def parse_caption(caption: str):
    """('Table 7', 'OV1') from 'Table 7: Overview ... (OV1)'."""
    num = re.search(r"Table\s+\d+[A-Za-z]?", caption)
    number = num.group(0) if num else ""
    parens = re.findall(r"\(([^)]+)\)", caption)
    parens = [p.strip() for p in parens if p.strip().lower() != "continued"]
    code = parens[-1] if parens else ""
    return number, code


def units_in(text: str):
    return [name for name, pat in UNIT_TOKENS if re.search(pat, text)]


def periods_in(text: str):
    seen, out = set(), []
    for m in DATE_RE.findall(text):
        if m not in seen:
            seen.add(m); out.append(m)
    return out


def _meta(source_page, content_type, number="", code="", text=""):
    """Chroma needs scalar metadata, so lists are comma-joined strings."""
    return {
        "source_page": int(source_page),
        "content_type": content_type,
        "table_number": number,
        "table_code": code,
        "units": ",".join(units_in(text)),
        "reporting_periods": ",".join(periods_in(text)),
    }


# --- table rendering ------------------------------------------------------
def _col_display(c):
    return c["header"] + (f" ({c['unit']})" if c["unit"] else "")


def _row_md(r, cols):
    cells = [r.get("ref", ""), r.get("label", "")] + [r["values"].get(c["header"], "") for c in cols]
    return "| " + " | ".join(cells) + " |"


def render_table_chunks(table, page):
    cols = table["columns"]
    rows = table["rows"]
    number, code = parse_caption(table["caption"])
    periods = periods_in(" ".join(c["header"] for c in cols))

    header_row = "| " + " | ".join(["Ref", "Item"] + [_col_display(c) for c in cols]) + " |"
    sep_row = "| " + " | ".join(["---"] * (len(cols) + 2)) + " |"
    has_units = any(c["unit"] for c in cols)
    units_row = ("| " + " | ".join(["", ""] + [f"({c['unit']})" if c["unit"] else "" for c in cols]) + " |"
                 if has_units else None)

    def preamble(continued=False, section=None):
        # caption already reads e.g. "Table 7: Overview ... (OV1)"; don't duplicate it.
        title = table["caption"] + (" — (continued)" if continued else "")
        code_tag = f"{number} {code}".strip()
        lines = [title,
                 f"[source page {page} | table | {code_tag} | reporting periods: "
                 f"{', '.join(periods) if periods else 'n/a'}]"]
        if continued and section:
            lines.append(f"(section: {section})")
        return "\n".join(lines)

    def header_block(continued=False, section=None):
        parts = [preamble(continued, section), "", header_row, sep_row]
        if units_row:
            parts.append(units_row)
        return "\n".join(parts)

    # Keep the whole table in one chunk unless it is very large.
    body_lines = [_row_md(r, cols) for r in rows]
    whole = header_block() + "\n" + "\n".join(body_lines)
    if len(whole) <= MAX_TABLE_CHARS:
        return [Document(page_content=whole,
                         metadata=_meta(page, "table", number, code, whole))]

    # Large table -> row groups, repeating the header block (and carrying the
    # most recent section banner so units context is never lost).
    chunks, group, section = [], [], None
    base_len = len(header_block())
    for r, line in zip(rows, body_lines):
        if r.get("is_section") and not r.get("values"):
            section = r.get("label") or section
        if group and base_len + sum(len(x) for x in group) + len(line) > MAX_TABLE_CHARS:
            text = header_block(continued=bool(chunks), section=section if chunks else None) + "\n" + "\n".join(group)
            chunks.append(Document(page_content=text,
                                   metadata=_meta(page, "table", number, code, text)))
            group = []
        group.append(line)
    if group:
        text = header_block(continued=bool(chunks), section=section if chunks else None) + "\n" + "\n".join(group)
        chunks.append(Document(page_content=text,
                               metadata=_meta(page, "table", number, code, text)))
    return chunks


# --- narrative chunking ---------------------------------------------------
def chunk_narrative(prose, page, splitter):
    out = []
    for piece in splitter.split_text(prose):
        piece = piece.strip()
        if piece:
            out.append(Document(page_content=f"[source page {page} | narrative]\n{piece}",
                                metadata=_meta(page, "narrative", text=piece)))
    return out


def build_chunks():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=NARRATIVE_SIZE, chunk_overlap=NARRATIVE_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    table_chunks, narrative_chunks = [], []
    for f in sorted(glob.glob(f"{EXTRACTED_DIR}/page-*.json")):
        d = json.load(open(f, encoding="utf-8"))
        page = d["page"]
        for t in d["tables"]:
            table_chunks.extend(render_table_chunks(t, page))
        if d.get("prose", "").strip():
            narrative_chunks.extend(chunk_narrative(d["prose"], page, splitter))
    return table_chunks, narrative_chunks


# --- store ----------------------------------------------------------------
def build_store(chunks):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_chroma import Chroma

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    # Reset ONLY the structured collection; never touch chroma_db/ as a whole
    # (that would destroy the baseline collection).
    Chroma(collection_name=COLLECTION, embedding_function=embeddings,
           persist_directory=PERSIST_DIR).delete_collection()
    store = Chroma(collection_name=COLLECTION, embedding_function=embeddings,
                   persist_directory=PERSIST_DIR,
                   collection_metadata={"hnsw:space": DISTANCE})

    # Gemini free embedding tier ~100 req/min -> throttled batches + 429 retry.
    EMBED_BATCH, PAUSE_S = 50, 61
    def add_with_retry(batch, tries=5):
        for _ in range(tries):
            try:
                store.add_documents(batch); return
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    print("  rate-limited; sleeping 60s ..."); time.sleep(60)
                else:
                    raise
        raise RuntimeError("embedding failed after retries")

    for i in range(0, len(chunks), EMBED_BATCH):
        add_with_retry(chunks[i:i + EMBED_BATCH])
        done = min(i + EMBED_BATCH, len(chunks))
        print(f"  embedded {done}/{len(chunks)} chunks")
        if done < len(chunks):
            time.sleep(PAUSE_S)
    return store


def main():
    load_dotenv(dotenv_path=".env")
    assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY missing (.env)"

    table_chunks, narrative_chunks = build_chunks()
    chunks = table_chunks + narrative_chunks
    print(f"built {len(chunks)} chunks: {len(table_chunks)} table, "
          f"{len(narrative_chunks)} narrative")

    store = build_store(chunks)
    print(f"persisted {store._collection.count()} vectors to "
          f"{PERSIST_DIR}/ (collection='{COLLECTION}', distance={DISTANCE})")

    # Confirm the baseline collection still coexists, untouched.
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    base = Chroma(collection_name=BASELINE_COLLECTION,
                  embedding_function=GoogleGenerativeAIEmbeddings(model=EMBED_MODEL),
                  persist_directory=PERSIST_DIR)
    print(f"baseline collection '{BASELINE_COLLECTION}' still present: "
          f"{base._collection.count()} vectors")


if __name__ == "__main__":
    main()
