"""Pre-generate the example concept maps and chapter timelines so they are served from cache.

    python scripts/prewarm_cache.py

Run once on the server after deploying (or locally, then copy artifacts/generated/). Cached objects are
served without any AI call, so these keep working even after the daily allowance is used up.
Costs a small, one-off number of calls; it honours the same daily budget as the app.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import learning_extractors as ex
from src.corpus import load_corpus
from src.generation import GenerationUnavailable

CONCEPTS = [("Prime Minister", "concept"), ("Canadarm", "concept"), ("Confederation", "concept"),
            ("Federal elections", "concept"), ("Sir John A. Macdonald", "person")]


def main() -> None:
    corpus = load_corpus()
    jobs = [(f"concept map: {n}", lambda n=n, k=k: ex.build_concept_hub(n, corpus, kind=k)) for n, k in CONCEPTS]
    jobs += [(f"timeline: {c}", lambda c=c: ex.build_timeline(corpus, chapter=c)) for c in corpus.chapters]
    for label, run in jobs:
        try:
            _, cached = run()
            print(f"{'cached ' if cached else 'built  '} {label}")
        except GenerationUnavailable as exc:
            print(f"stopped at '{label}': {exc}")
            return
        except Exception as exc:                       # keep going; one bad topic should not stop the rest
            print(f"failed  {label}: {type(exc).__name__}")


if __name__ == "__main__":
    main()
