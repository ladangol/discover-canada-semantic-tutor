"""Build artifacts/fact_bank.json: "Did you know?" facts, each tied to a verbatim quote in the guide.

One-time, offline (about 60 batched LLM calls through src/generation.py, so the usual key/budget rules apply). The model proposes facts; code
keeps one only if its quote is verbatim in the passage it cites, every number in the fact is in that quote, every capitalised word is in the
passage, and it is not a near-duplicate of a fact already kept (the guide's passages overlap, so the same sentence can come up twice).
A second, targeted pass fills in any province or territory (and money) that the first pass left without a fact.
The 14 hand-written facts in src/facts.py are never changed; this file only adds to them.   python scripts/build_fact_bank.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import facts, tutor
from src.config import generation_model
from src.corpus import load_corpus
from src.embeddings import get_encoder
from src.generation import generate
from src.learning_validation import norm

BATCH = 4
WORKERS = 6
DUPLICATE_SIMILARITY = 0.80
PLACES = ["Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland and Labrador", "Northwest Territories", "Nova Scotia",
          "Nunavut", "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan", "Yukon"]
MONEY = re.compile(r"\$\s?\d|\bcoin\b|\bbill\b|\bcurrency\b", re.I)

SYSTEM = """You write "Did you know?" facts for learners using the Canadian citizenship guide Discover Canada. You are given numbered passages.
For each passage that has one, write up to 2 short, surprising or memorable facts that the passage itself states. Rules:
- Use ONLY what the passage says. Never add outside knowledge, dates, names or numbers.
- "fact" completes the sentence "Did you know …": start lower-case (not with "that"), one sentence, under 200 characters, ending with "?".
  Example: "New Brunswick is the only officially bilingual province?"
- "quote" is copied EXACTLY (4-30 words) from that passage and supports the whole fact.
- Skip dry procedure, definitions and anything not interesting on its own; an empty list for a passage is fine.
Output JSON only: {"facts": [{"source": "P1", "fact": "...?", "quote": "..."}]}"""


def _block(label: str, row) -> str:
    return f"[{label}] ({row.section}, page {row.page})\n{row.body}"


def _ask(rows, hint: str = "") -> list[dict]:
    labelled = {f"P{n}": r for n, r in enumerate(rows, start=1)}
    user = ("\n\n".join(_block(k, r) for k, r in labelled.items())
            + (f"\n\nFocus: {hint}" if hint else ""))
    try:
        data = tutor._ask_llm_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                   lambda m, **kw: generate(m, temperature=0.2, **kw), max_tokens=900)
    except Exception as exc:                                      # keep going; the count is reported at the end
        print(f"  call failed: {type(exc).__name__}", file=sys.stderr)
        return []
    out = []
    for item in (data.get("facts", []) if isinstance(data, dict) else []):
        if isinstance(item, dict) and (row := labelled.get(str(item.get("source", "")).strip().upper())) is not None:
            out.append({"row": row, "fact": " ".join(str(item.get("fact", "")).split()), "quote": " ".join(str(item.get("quote", "")).split())})
    return out


def problem(row, fact: str, quote: str) -> str:
    """Why a proposed fact is rejected ('' = fine). Everything here is checked by code, not by the model."""
    if not (25 <= len(fact) <= 220) or not fact.endswith("?"):
        return "not a single sentence ending in ?"
    if fact.lower().startswith(("did you know", "that ")):
        return "starts with the lead-in"
    if len(norm(quote).split()) < 4 or norm(quote) not in norm(row.body):
        return "quote is not verbatim in the passage"
    if set(re.findall(r"\d[\d,.]*\d|\d", fact)) - set(re.findall(r"\d[\d,.]*\d|\d", quote)):
        return "a number in the fact is not in the quote"
    for word in re.findall(r"(?<=[a-z,] )[A-ZÉ][\w'’-]+", fact):
        if norm(word) and norm(word) not in norm(row.body):
            return f"“{word}” is not in the passage"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N passages (testing)")
    a = ap.parse_args()
    corpus = load_corpus()
    rows = [r for r in corpus.chunks.itertuples() if r.word_count >= tutor.MIN_QUIZ_WORDS]
    rows = rows[: a.limit] if a.limit else rows

    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    with ThreadPoolExecutor(WORKERS) as pool:
        proposals = [p for ps in pool.map(_ask, batches) for p in ps]

    kept: list[dict] = []
    rejected: dict[str, int] = {}
    vectors = list(get_encoder().encode([f.text for f in facts.HAND_WRITTEN], normalize_embeddings=True, show_progress_bar=False))

    def accept(proposal: dict) -> bool:
        why = problem(proposal["row"], proposal["fact"], proposal["quote"])
        if not why:
            v = get_encoder().encode([proposal["fact"]], normalize_embeddings=True, show_progress_bar=False)[0]
            if any(float(v @ w) > DUPLICATE_SIMILARITY for w in vectors):
                why = "near-duplicate of a fact already kept"
            else:
                vectors.append(v)
        if why:
            key = "a capitalised word is not in the passage" if why.startswith("“") else why
            rejected[key] = rejected.get(key, 0) + 1
            return False
        r = proposal["row"]
        kept.append({"text": proposal["fact"], "page": r.page, "section": r.section, "quote": proposal["quote"]})
        return True

    for p in proposals:
        accept(p)

    # targeted pass: a place (or money) with no fact yet gets its own passages and a focus line
    def covered(label: str) -> bool:
        return any(label.lower() in (k["section"] + " " + k["text"]).lower() for k in kept) or \
               any(label.lower() in (f.section + " " + f.text).lower() for f in facts.HAND_WRITTEN)
    gaps = [(p, f"a fact about {p}", [r for r in rows if r.section == p or r.section.startswith(p)]) for p in PLACES if not covered(p)]
    if not any(MONEY.search(k["quote"]) for k in kept):
        gaps.append(("money", "a fact about Canadian money (bills, coins, who is pictured)", [r for r in rows if MONEY.search(r.body)]))
    for label, hint, pool_rows in gaps:
        for p in _ask(pool_rows[:BATCH], hint):
            accept(p)

    path = facts.BANK_PATH
    path.write_text(json.dumps({"version": 1, "model": generation_model(),
                                "stats": {"passages": len(rows), "proposed": len(proposals), "kept": len(kept), "rejected": rejected,
                                          "targeted": [g[0] for g in gaps]},
                                "facts": kept}, ensure_ascii=False, indent=1))
    print(f"passages {len(rows)} · proposed {len(proposals)} · kept {len(kept)} · rejected {rejected} · targeted {[g[0] for g in gaps]} -> {path}")
    for place in PLACES:
        n = sum(place.lower() in (k["section"] + " " + k["text"]).lower() for k in kept) + \
            sum(place.lower() in (f.section + " " + f.text).lower() for f in facts.HAND_WRITTEN)
        print(f"  {place}: {n}")


if __name__ == "__main__":
    main()
