"""Build artifacts/suggestion_bank.json: 2 learner-style questions per passage, kept only if the frozen retrieval returns the passage.

One-time, offline (about 200 small LLM calls through src/generation.py, so the usual key/budget rules apply). Re-run only if the
corpus or the prompt changes.   python scripts/build_suggestion_bank.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import tutor
from src.config import generation_model
from src.corpus import load_corpus
from src.generation import generate
from src.retrieval import retrieve
from src.suggestions import BANK_PATH

SYSTEM = """You write short study questions for the Canadian citizenship guide Discover Canada. You are given ONE passage. Write exactly 2
natural questions a learner might type or say, each answerable from this passage alone: one starting with Who, What, When, Where, Why or
How about the main idea, and one about a different detail or key term in it. Use the passage's own names and terms; make each question
self-contained (never say "this passage" or "the text"); under 90 characters; each ends with "?".
Output JSON only: {"questions": ["...?", "...?"]}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N passages (testing)")
    a = ap.parse_args()
    c = load_corpus()
    idx = [i for i, w in enumerate(c.chunks.word_count) if w >= tutor.MIN_QUIZ_WORDS]
    idx = idx[: a.limit] if a.limit else idx

    def gen_for(i: int):
        src = tutor.sources_from_corpus(c, [i])[0]
        try:
            data = tutor._ask_llm_json([{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": f"SECTION: {src.section}\n\nPASSAGE:\n{src.text}"}],
                                       generate, max_tokens=300)
        except Exception as exc:                       # keep going; report at the end
            return src, [], type(exc).__name__
        qs = [" ".join(str(q).split()) for q in (data.get("questions", []) if isinstance(data, dict) else [])]
        return src, [q for q in qs if q.endswith("?") and 10 <= len(q) <= 110], None

    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(gen_for, idx))

    kept, dropped, failed = [], 0, 0
    for src, qs, err in results:
        failed += err is not None
        for q in qs:
            if any(x.chunk_id == src.chunk_id for x in retrieve(q, k=5)):     # round trip through the frozen retrieval
                kept.append({"q": q, "chunk_id": src.chunk_id, "section": src.section, "page": src.page})
            else:
                dropped += 1
    seen, uniq = set(), []
    for b in kept:
        if b["q"].lower() not in seen:
            seen.add(b["q"].lower())
            uniq.append(b)
    BANK_PATH.write_text(json.dumps({"version": 1, "model": generation_model(),
                                     "stats": {"passages": len(idx), "kept": len(uniq), "dropped_roundtrip": dropped, "failed_calls": failed},
                                     "questions": uniq}, ensure_ascii=False, indent=1))
    print(f"passages {len(idx)} · kept {len(uniq)} · dropped (own passage not in top-5) {dropped} · failed calls {failed} -> {BANK_PATH}")


if __name__ == "__main__":
    main()
