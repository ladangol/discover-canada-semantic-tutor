"""Document-wide enumeration: "What languages are mentioned in the guide?"

This is a SEPARATE evidence-gathering workflow. It is not Q&A and it does not touch the frozen, benchmarked retriever (Variant A, k=5).
Top-5 semantic retrieval cannot answer "list everything" questions: the passages that hold the items rank 9th, 40th or 78th for a
question like that. Instead this workflow READS passages exhaustively and lists what they say:

  scope     quick    only passages that contain the learner's own word (a literal stem match, e.g. "languag"; no synonyms), or
            thorough every passage in the guide;
  extract   a model reads each batch and returns items with a verbatim quote;
  validate  the source label is real, the quote is verbatim in that passage, the item is named in it, and the guide's own wording is checked;
  group     items the guide itself calls the requested kind are listed first; items it words differently ("dialect", "word") are listed
            SEPARATELY with the guide's word, never silently merged into the kind;
  report    how many passages were read, how many AI calls it took, and what could not be verified.

No synonym or category mapping is used anywhere: "First Nations", "Métis", "Inuit" and "Aboriginal" stay distinct (a focus on one of the
three peoples keeps only items whose passage ties them to it, via concept_semantics.scope_conflict). Cost is explicit and shown first;
results are cached. Honest limit: it finds what the guide's wording makes findable; an item named without the kind's word or the
guide's related word near it can be missed, and the result says how it was scoped.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable

from src import concept_semantics as semantics
from src import config, learning_cache
from src.corpus import Corpus
from src.generation import generate
from src.learning_extractors import _json_with_one_retry
from src.learning_validation import norm
from src.retrieval import RetrievedChunk
from src.tutor import Generate, format_sources, sources_from_corpus

ENUMERATION_PROMPT_VERSION = "e1"
BATCH = 8                       # passages per model call
WORKERS = 4
MODES = ("quick", "thorough")
MAX_KIND_CHARS = 40
STOP = {"the", "a", "an", "all", "every", "list", "of", "in", "guide", "discover", "canada", "mentioned", "names", "name"}


@dataclass(frozen=True)
class Evidence:
    quote: str
    page: str
    section: str
    chunk_id: str


@dataclass(frozen=True)
class EnumItem:
    name: str
    stated_as: str                         # the word the GUIDE uses for it ("language", "dialect", "word", …)
    exact: bool                            # True: the guide itself calls it the requested kind
    evidence: list[Evidence] = field(default_factory=list)


@dataclass(frozen=True)
class ScanPlan:
    kind: str
    mode: str
    passages: int                          # how many passages will be read
    calls: int                             # how many AI calls that takes
    scope: str                             # what "quick" or "thorough" means here, in words


@dataclass(frozen=True)
class EnumerationResult:
    kind: str
    focus: str
    mode: str
    scanned: int
    total_passages: int
    calls: int
    items: list[EnumItem]
    omitted: list[str] = field(default_factory=list)

    @property
    def exact(self) -> list[EnumItem]:
        return [i for i in self.items if i.exact]

    @property
    def worded_differently(self) -> list[EnumItem]:
        return [i for i in self.items if not i.exact]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EnumerationResult":
        items = [EnumItem(**{**i, "evidence": [Evidence(**e) for e in i["evidence"]]}) for i in d["items"]]
        return cls(**{**d, "items": items})


# --- the learner's words -------------------------------------------------------------------------------------------------

def clean_kind(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'’-]", " ", text or "")).strip()[:MAX_KIND_CHARS]


def head_stem(kind: str) -> str:
    """The stem of the learner's own head word ("languages" -> "langua"). No synonyms: the same word, any ending."""
    words = [w for w in re.findall(r"[\wÀ-ÿ'’-]+", kind.lower()) if w not in STOP]
    return words[-1][:6] if words else ""


def passage_has_word(text: str, stem: str) -> bool:
    return bool(stem) and re.search(rf"\b{re.escape(stem)}", text, re.I) is not None


# --- plan and run --------------------------------------------------------------------------------------------------------------

def _indices(corpus: Corpus, kind: str, mode: str) -> list[int]:
    if mode == "thorough":
        return list(range(len(corpus.chunks)))
    stem = head_stem(kind)
    return [i for i, t in enumerate(corpus.chunks.text) if passage_has_word(t, stem)]


def plan_scan(corpus: Corpus, kind: str, mode: str = "quick") -> ScanPlan:
    kind, mode = clean_kind(kind), mode if mode in MODES else "quick"
    n = len(_indices(corpus, kind, mode))
    scope = (f"every passage in the guide ({len(corpus.chunks)})" if mode == "thorough"
             else f"the passages that contain the word “{head_stem(kind)}…” (a literal match, no synonyms)")
    return ScanPlan(kind, mode, n, -(-n // BATCH) if n else 0, scope)


def _prompt(kind: str, focus: str, sources: list[RetrievedChunk]) -> str:
    return f"""KIND: {kind}
FOCUS: {focus or "(none)"}

SOURCES:
{format_sources(sources)}

List every item that a source itself names as a "{kind}", or unmistakably uses as one. Rules:
- Use ONLY the source text. Never add outside knowledge and never widen the category: a people, a place, a person or a single word is not a
  "{kind}" unless the source says so. If the source words it differently (for example "dialect" or "word"), still list it but put the
  source's own word in "stated_as".
- "stated_as" is the word the SOURCE uses for the item (copy it). "evidence_quote" is copied EXACTLY (3-25 words) from one source and
  contains the item's name or the word that names its kind.
- If FOCUS names a people (First Nations, Métis, Inuit), keep only items the source ties to that people. Never treat "Aboriginal" or
  "Indigenous" as the same as any of them.
- Return JSON only: {{"items": [{{"name": str, "stated_as": str, "source": "S1", "evidence_quote": str}}]}}. An empty list is correct when there are none."""


def _focus_problem(focus: str, name: str, quote: str, chunk_text: str) -> str:
    """With a focus on First Nations, Métis or Inuit the passage must tie the item to THAT people: naming another people excludes it, and
    naming none is not enough ("keep only items the source ties to that people")."""
    if conflict := semantics.scope_conflict(focus, name, quote, chunk_text):
        return conflict
    want = semantics.scopes_in(focus)
    if want and not (want & semantics.nearest_scopes(name, quote, chunk_text)):
        return f"its passage does not tie it to the {', '.join(sorted(want))}"
    return ""


def _validate(raw, kind: str, focus: str, by_label: dict[str, RetrievedChunk]) -> tuple[list[tuple[str, str, bool, Evidence]], list[str]]:
    out, omitted = [], []
    stem = head_stem(kind)
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        name, stated = str(item.get("name", "")).strip(), str(item.get("stated_as", "")).strip().lower()
        label = name or "(unnamed)"
        chunk = by_label.get(str(item.get("source", "")).strip().upper())
        quote = str(item.get("evidence_quote", "")).strip()
        if not name or chunk is None:
            omitted.append(f"{label}: missing name or an unknown source")
        elif len(norm(quote).split()) < 3:
            omitted.append(f"{label}: the quote is too short to check (needs at least 3 words)")
        elif norm(quote) not in norm(chunk.text):
            omitted.append(f"{label}: quote is not in the cited passage")
        elif not (semantics.mentions(name, quote) or semantics.mentions(name, chunk.text)):
            omitted.append(f"{label}: not named in the cited passage")
        elif not (passage_has_word(quote, stem) or (stated and stated in quote.lower())):
            omitted.append(f"{label}: the quote uses neither “{kind}” nor the guide's own word for it")
        elif focus and (problem := _focus_problem(focus, name, quote, chunk.text)):
            omitted.append(f"{label}: {problem}")
        else:
            # decided from the QUOTE, not from the model's label: it counts as the requested kind only if the quote itself holds the kind's word
            # AND names the item ("the Inuktitut language"); otherwise it is listed separately, with the guide's own word when the quote has one
            exact = passage_has_word(quote, stem) and semantics.mentions(name, quote)
            own_word = stated if (stated and stated in quote.lower() and stated != name.lower()) else ""
            out.append((name, kind.lower() if exact else own_word, exact, Evidence(quote, chunk.page, chunk.section, chunk.chunk_id)))
    return out, omitted


def _key(name: str, stem: str = "") -> str:
    """Merge key: "the Inuktitut", "Inuktitut language" and "Inuktitut" are one item (the kind's own word and articles are dropped)."""
    words = [w for w in norm(name).split() if w not in {"the", "a", "an"} and not (stem and w.startswith(stem))]
    return " ".join(words) or norm(name)


def _group(rows: list[tuple[str, str, bool, Evidence]], stem: str = "") -> list[EnumItem]:
    """One item per name: quotes and pages merged. An item counts as the requested kind if the guide words ANY mention that way (every quote is kept and shown)."""
    merged: dict[str, dict] = {}
    for name, stated, exact, ev in rows:
        m = merged.setdefault(_key(name, stem), {"name": name, "stated": stated, "exact": False, "ev": []})
        m["exact"] |= exact
        if exact or not m["stated"]:
            m["stated"] = stated if (exact or not m["exact"]) else m["stated"]
        m["name"] = min(m["name"], name, key=len)                    # the shortest form: "Chinese", not "Chinese languages"
        if not any(e.quote == ev.quote for e in m["ev"]):
            m["ev"].append(ev)
    items = [EnumItem(m["name"], m["stated"], m["exact"], sorted(m["ev"], key=lambda e: (len(e.page), e.page))) for m in merged.values()]
    return sorted(items, key=lambda i: (not i.exact, (i.evidence[0].page.zfill(3) if i.evidence else "999"), i.name.casefold()))


def scan_guide(kind: str, corpus: Corpus, *, focus: str = "", mode: str = "quick", use_cache: bool = True, gen: Generate = generate,
               progress: Callable[[int, int], None] | None = None) -> tuple[EnumerationResult, bool]:
    """Read the guide for `kind`. Returns (result, loaded_from_cache). Separate from Q&A; costs `plan_scan(...).calls` AI calls."""
    kind, focus, mode = clean_kind(kind), clean_kind(focus), mode if mode in MODES else "quick"
    if not head_stem(kind):
        raise ValueError("Say what you want listed, for example: languages, provinces, prime ministers.")
    params = {"kind": "enumeration", "model": config.generation_model(), "prompt_version": ENUMERATION_PROMPT_VERSION, "corpus_chunks": len(corpus),
              "what": kind.casefold(), "focus": focus.casefold(), "mode": mode}
    path = learning_cache.cache_path("enumeration", f"{kind}_{mode}", params)
    if use_cache and (cached := learning_cache.load(path)):
        return EnumerationResult.from_dict(cached), True

    idx = _indices(corpus, kind, mode)
    batches = [sources_from_corpus(corpus, idx[i:i + BATCH]) for i in range(0, len(idx), BATCH)]
    done = [0]

    def read(sources: list[RetrievedChunk]):
        data = _json_with_one_retry([{"role": "system", "content": "You read passages of the Canadian citizenship guide Discover Canada and extract facts "
                                      "exactly as written. Output a single JSON object and nothing else."},
                                     {"role": "user", "content": _prompt(kind, focus, sources)}],
                                    lambda messages, **kw: gen(messages, temperature=0.0, **kw))         # extraction, not creativity: as repeatable as the model allows
        rows, omitted = _validate(data.get("items") if isinstance(data, dict) else None, kind, focus, {s.source_id: s for s in sources})
        done[0] += 1
        if progress:
            progress(done[0], len(batches))
        return rows, omitted

    rows, omitted = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for r, o in pool.map(read, batches):
            rows += r
            omitted += o
    result = EnumerationResult(kind, focus, mode, len(idx), len(corpus.chunks), len(batches), _group(rows, head_stem(kind)), omitted)
    learning_cache.save(path, params, result.to_dict())
    return result, False
