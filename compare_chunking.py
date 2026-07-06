#!/usr/bin/env python3
"""Before/after: same 10 stress-test questions, same BASELINE pipeline, two stores.

Runs the identical non-agentic pipeline (top-k=4, gpt-4o-mini @ temp 0, identical
grounding prompt) against `pillar3_baseline` (naive chunks) and `pillar3_structured`
(structure-aware chunks). No grading/routing/rewriting/web search. Writes
`data/chunking_comparison.json` for the write-up.
"""
from __future__ import annotations
import json
import os
import time

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

PERSIST_DIR = "chroma_db"
EMBED_MODEL = "models/gemini-embedding-001"
GEN_MODEL = "gpt-4o-mini"
TOP_K = 4
COLLECTIONS = ["pillar3_baseline", "pillar3_structured"]
REFUSAL = "does not contain enough information"

# Identical prompt + context format to baseline_rag.ipynb (parity is the point).
PROMPT = ChatPromptTemplate.from_template(
    "You are a banking regulatory-disclosure assistant. Answer the QUESTION using "
    "ONLY the CONTEXT extracted from HSBC's Pillar 3 disclosure.\n"
    "Rules:\n"
    "- Use only facts present in the CONTEXT. If the figure, its reporting date, or "
    "its unit is not in the CONTEXT, reply exactly: 'The retrieved context does not "
    "contain enough information to answer.'\n"
    "- When you state a figure, include its unit ($m, $bn, %, bps) and its reporting "
    "date exactly as shown.\n\n"
    "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
)

def format_context(hits):
    return "\n\n".join(
        f"[chunk {i+1} | page {h.metadata['source_page']}]\n{h.page_content}"
        for i, h in enumerate(hits)
    )

# Post-hoc scoring spec (same heuristic + tokens as the baseline notebook).
SCORING = {
 1:  dict(primary="14.9", units=["%"], nums=[["14.9"]], parts=[["14.9"]]),
 2:  dict(primary="132.6", units=["bn"], nums=[["132.6"]], parts=[["132.6"]]),
 3:  dict(primary="5.3", units=["%"], nums=[["5.3"]],
         parts=[["5.3"], ["down", "fell", "fall", "decreas", "lower", "reduc"]]),
 4:  dict(primary="38,490", units=["m", "bn"], nums=[["80,870", "80.9", "80,870m"]],
         parts=[["80,870", "80.9"]]),
 5:  dict(primary="end-point", units=[],
         nums=[["same", "identical", "both", "no difference", "unchanged"]],
         parts=[["transitional"], ["end-point", "end point"]]),
 6:  dict(primary="137", units=["%", "bn"], nums=[["137"], ["143"], ["702"]],
         parts=[["137"], ["143"], ["702"]]),
 7:  dict(primary="110", units=["bps"], nums=[["110"]],
         parts=[["110"], ["january 2026", "2026"]]),
 8:  dict(primary="8%", units=["%"], nums=[["8%", "8 %"]],
         parts=[["8%", "8 %"], ["article 92", "92(1)", "92 (1)"]]),
 9:  dict(primary="888,647", units=["m", "bn"], nums=[["888,647", "888.6"]],
         parts=[["888,647", "888.6"], ["dollar", "usd", "us$"]]),
 10: dict(primary="120,716", units=["m", "bn"],
          nums=[["120,716"], ["106,472"], ["14,244", "14.2"]],
          parts=[["120,716"], ["106,472"], ["14,244", "14.2"]]),
}

def has_any(t, toks): return any(x.lower() in t.lower() for x in toks)
def has_group(t, groups): return all(has_any(t, g) for g in groups)

def score(qid, answer, ctx):
    s = SCORING[qid]
    answered = REFUSAL.lower() not in answer.lower()
    retrieved_has = s["primary"].lower() in ctx.lower()
    numerically_ok = answered and has_group(answer, s["nums"])
    complete = answered and has_group(answer, s["parts"])
    unit_ok = (not s["units"]) or (answered and has_any(answer, s["units"]))
    grounded = (not answered) or (numerically_ok and retrieved_has)
    return dict(
        answered=answered, retrieved_has_fact=retrieved_has, grounded=grounded,
        numerically_correct=numerically_ok, complete=complete, unit_ok=unit_ok,
        figure_right_unit_wrong=(numerically_ok and not unit_ok),
        passed=(grounded and numerically_ok and complete and unit_ok),
    )

def main():
    load_dotenv(dotenv_path=".env")
    questions = json.load(open("eval_set.json"))["questions"]
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    llm = ChatOpenAI(model=GEN_MODEL, temperature=0)

    def answer_against(store, q):
        for _ in range(5):
            try:
                hits = store.as_retriever(search_kwargs={"k": TOP_K}).invoke(q)
                ctx = format_context(hits)
                ans = llm.invoke(PROMPT.format_messages(context=ctx, question=q)).content.strip()
                return ans, [h.metadata["source_page"] for h in hits], ctx
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    print("  rate-limited; sleeping 60s ..."); time.sleep(60)
                else:
                    raise
        raise RuntimeError("retrieval/generation failed after retries")

    results = {}
    for coll in COLLECTIONS:
        store = Chroma(collection_name=coll, embedding_function=embeddings,
                       persist_directory=PERSIST_DIR)
        print(f"\n=== {coll} ({store._collection.count()} vectors) ===")
        rows = []
        for item in questions:
            ans, pages, ctx = answer_against(store, item["question"])
            sc = score(item["id"], ans, ctx)
            rows.append(dict(id=item["id"], question=item["question"], gold=item["gold"],
                             trap=item["trap"], answer=ans, pages=pages, **sc))
            print(f"Q{item['id']:>2} {'PASS' if sc['passed'] else 'FAIL'} | pages {pages}\n    {ans}")
            time.sleep(1)
        results[coll] = rows

    json.dump(results, open("data/chunking_comparison.json", "w"), indent=2, ensure_ascii=False)
    b = sum(r["passed"] for r in results["pillar3_baseline"])
    s = sum(r["passed"] for r in results["pillar3_structured"])
    print(f"\nPASS  baseline {b}/10   ->   structured {s}/10")
    print("saved data/chunking_comparison.json")

if __name__ == "__main__":
    main()
