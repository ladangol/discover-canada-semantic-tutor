"""List everything of one kind that the guide mentions, by reading passages (NOT top-5 retrieval). See src/enumeration.py.

    python scripts/scan_guide.py languages                      # quick: passages that contain your word (3 AI calls)
    python scripts/scan_guide.py languages --mode thorough      # every passage (about 30 AI calls)
    python scripts/scan_guide.py languages --focus "First Nations"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import enumeration as en
from src.corpus import load_corpus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", help="what to list, in your own words: languages, provinces, prime ministers …")
    ap.add_argument("--focus", default="", help="optionally one people: First Nations, Métis or Inuit")
    ap.add_argument("--mode", choices=en.MODES, default="quick")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    corpus = load_corpus()
    plan = en.plan_scan(corpus, a.kind, a.mode)
    print(f"Plan: read {plan.passages} passages ({plan.scope}) in {plan.calls} AI calls.")
    res, cached = en.scan_guide(a.kind, corpus, focus=a.focus, mode=a.mode, use_cache=not a.no_cache,
                                progress=lambda d, t: print(f"  {d}/{t} batches", end="\r"))
    print(f"\n{'(from cache) ' if cached else ''}Read {res.scanned} of {res.total_passages} passages in {res.calls} AI calls.\n")
    print(f"Guide words it as “{res.kind}”:")
    for i in res.exact:
        print(f"  • {i.name}   pages {', '.join(sorted({e.page for e in i.evidence}))}   “{i.evidence[0].quote}”")
    if not res.exact:
        print("  (nothing)")
    print("\nMentioned, but the guide words it differently:")
    for i in res.worded_differently:
        print(f"  • {i.name}   the guide says “{i.stated_as or '…'}”   pages {', '.join(sorted({e.page for e in i.evidence}))}   “{i.evidence[0].quote}”")
    if not res.worded_differently:
        print("  (nothing)")
    if res.omitted:
        print(f"\n{len(res.omitted)} item(s) dropped by validation:")
        for o in res.omitted:
            print("  –", o)


if __name__ == "__main__":
    main()
