"""Evidence -> LLM -> strict JSON -> validation, for Concept Hub and Timeline.

Pipeline (each stage is separate):
  retrieval   : embeddings pick WHICH chunks to inspect (retrieval.retrieve, or a whole
                chapter/section from the corpus). This is candidate gathering for a
                visualization, not the frozen k=5 answer configuration.
  extraction  : the LLM turns those source passages into JSON (this module).
  validation  : learning_validation drops anything without source support.
  rendering   : visualization.py.
"""
from __future__ import annotations


from concurrent.futures import ThreadPoolExecutor

from src import concept_semantics as semantics, config, learning_cache
from src.corpus import Corpus
from src.generation import GenerationUnavailable, generate
from src.learning_models import ConceptHub, TimelineData, TimelineEvent
from src.learning_validation import (
    merge_duplicate_events,
    normalize_category,
    validate_events,
    validate_hub_nodes,
    validate_summary,
)
from src.retrieval import RetrievedChunk, retrieve
from src.tutor import Generate, QuizGenerationError, extract_json, format_sources, sources_from_corpus

MAX_TOKENS = 3500

EXTRACT_SYSTEM = """You convert supplied SOURCE passages from the Canadian citizenship guide
Discover Canada into structured JSON. You use ONLY facts stated in the SOURCE passages. You never
use outside knowledge, never fill in missing dates or relationships, and you omit anything the
sources do not state. Output a single JSON object and nothing else."""


def _json_with_one_retry(messages: list[dict[str, str]], gen: Generate):
    """Parse the model's JSON; on failure ask once more, then give up."""
    last = ""
    for attempt in range(2):
        reply = gen(messages if attempt == 0 else messages + [
            {"role": "assistant", "content": last[:1500]},
            {"role": "user", "content": "That was not valid JSON. Reply again with the JSON object only."},
        ], max_tokens=MAX_TOKENS).text
        try:
            return extract_json(reply)
        except ValueError:
            last = reply
    raise QuizGenerationError("The model did not return valid JSON. Try again.")


def _params(kind: str, request: dict) -> dict:
    return {"kind": kind, "model": config.generation_model(),
            "prompt_version": config.LEARNING_PROMPT_VERSION, "corpus_chunks": None, **request}


# =============================================================================
# Concept hub / person hub
# =============================================================================

def _hub_prompt(concept: str, kind: str, sources: list[RetrievedChunk], max_nodes: int) -> str:
    what = ("a PERSON: include only facts the sources state about this person (role, events, "
            "relationships). Do not write a biography." if kind == "person"
            else "a CONCEPT: include related people, institutions, events and ideas that the sources "
                 "connect to it.")
    types, predicates = semantics.prompt_vocabulary()
    return f"""CENTRAL {kind.upper()}: {concept}
This is {what}

SOURCES:
{format_sources(sources)}

Build a hub-and-spoke learning map. Return JSON:
{{"central_concept": str,
  "summary": str (1-2 sentences, only what the sources say),
  "summary_sources": ["S1"],
  "central_entity_type": one of: {types},
  "nodes": [{{"label": str (short noun phrase),
             "entity_type": one of the same list (a people is not a language; a word is not a language family; an institution is not an event),
             "predicate": one of the names below, or "mentioned_with" (the sources only mention both together) or "other",
             "subject_is_center": true if the CENTRAL item is the SUBJECT of the predicate (centre -> node), false if the NODE is the subject (node -> centre),
             "relationship": str (1-4 words, ONLY when predicate is "other"; "[central item] <relationship> <node label>" must be true per the sources),
             "explanation": str (1-2 sentences, nothing the quote does not support),
             "sources": ["S2"],
             "evidence_quote": str (copied EXACTLY, 5-25 words, from one cited source, STATING the relationship)}}]}}

Predicates, each written subject -> object:
{predicates}

Rules: at most {max_nodes} nodes (fewer if the sources support fewer; an empty list is the right answer when nothing is supported); every
node needs sources and an exact evidence_quote that states the relationship; no duplicate nodes; never infer a relationship the text
does not state (living near, being mentioned together, or sharing a heading is NOT a relationship: use "mentioned_with"). First Nations, Métis
and Inuit are three distinct peoples: never attach a Métis or Inuit item to a First Nations centre, or the reverse, and never treat
"Aboriginal" or "Indigenous" as the same as any of them. A language belongs to a people only if the text says that people speak it.
Do not mention any year that is not in the cited source."""


def build_concept_hub(
    concept: str, corpus: Corpus, *, kind: str = "concept", max_nodes: int = 6,
    use_cache: bool = True, gen: Generate = generate,
) -> tuple[ConceptHub, bool]:
    """Returns (hub, loaded_from_cache)."""
    concept = concept.strip()
    request = {"concept": concept.casefold(), "hub_kind": kind, "max_nodes": max_nodes,
               "k": config.HUB_CANDIDATES_K}
    params = _params("concept_hub", request) | {"corpus_chunks": len(corpus), "prompt_version": config.HUB_PROMPT_VERSION}
    path = learning_cache.cache_path("person_hub" if kind == "person" else "concept_hub", concept, params)
    if use_cache and (cached := learning_cache.load(path)):
        return ConceptHub.from_dict(cached), True

    sources = retrieve(concept, k=config.HUB_CANDIDATES_K)
    by_label = {s.source_id: s for s in sources}
    data = _json_with_one_retry(
        [{"role": "system", "content": EXTRACT_SYSTEM},
         {"role": "user", "content": _hub_prompt(concept, kind, sources, max_nodes)}], gen)
    if not isinstance(data, dict):
        raise QuizGenerationError("The model returned an unexpected structure. Try again.")

    center_type = str(data.get("central_entity_type", "")).strip().casefold()
    center_type = center_type if center_type in semantics.ENTITY_TYPES else ("person" if kind == "person" else "")
    nodes, omitted = validate_hub_nodes(data.get("nodes"), concept, by_label, max_nodes, center_type)
    summary, summary_ids = validate_summary(data.get("summary"), data.get("summary_sources"), by_label, concept)
    hub = ConceptHub(concept, kind, summary, nodes, summary_ids, omitted,
                     central_category=semantics.CATEGORY_FOR_TYPE.get(center_type) or normalize_category(data.get("central_category")),
                     central_entity_type=center_type)
    learning_cache.save(path, params, hub.to_dict())
    return hub, False


# =============================================================================
# Timeline
# =============================================================================

def _timeline_prompt(title: str, sources: list[RetrievedChunk]) -> str:
    return f"""TIMELINE TOPIC: {title}

SOURCES:
{format_sources(sources)}

List EVERY dated event in these sources: include each event or milestone that has a year or date
(typically one entry per dated sentence), not just the most famous ones. Return JSON:
{{"events": [{{"date_label": str (as written in the source, e.g. "1812–1814", "1867"),
              "start_year": int or null, "end_year": int or null,
              "title": str (<= 8 words),
              "description": str (<= 30 words, only facts in the sources),
              "importance": str (one sentence ONLY if the sources state why it matters, else ""),
              "sources": ["S3"],
              "evidence_quote": str (copied EXACTLY, 5-25 words, from a cited source, containing the date or event)}}]}}

Rules: dates must appear in the cited source; never fill a missing year from general knowledge;
preserve ranges such as 1812–1814; if a source gives only a broad period with no year, use that
wording as date_label and set both years to null; one entry per event (merge duplicates); every event
needs sources and an exact evidence_quote."""


def _batches(corpus: Corpus, indices: list[int]) -> list[list[RetrievedChunk]]:
    size = config.TIMELINE_BATCH_CHUNKS
    return [sources_from_corpus(corpus, indices[i:i + size]) for i in range(0, len(indices), size)]


def build_timeline(
    corpus: Corpus, *, chapter: str | None = None, section: str | None = None,
    topic: str | None = None, use_cache: bool = True, cache_only: bool = False, gen: Generate = generate,
) -> tuple[TimelineData, bool]:
    """Timeline for a whole chapter, a section, or a topic query.

    Chapters/sections use every assigned chunk (in document order, batched);
    a topic uses the top candidate chunks from dense retrieval.
    """
    if topic:
        title, scope = f"Timeline: {topic.strip()}", {"topic": topic.strip().casefold()}
    elif section:
        title, scope = f"Timeline: {section}", {"chapter": chapter, "section": section}
    elif chapter:
        title, scope = f"Timeline: {chapter}", {"chapter": chapter}
    else:
        raise ValueError("Choose a chapter, section or topic.")
    request = scope | {"k": config.TIMELINE_CANDIDATES_K, "batch": config.TIMELINE_BATCH_CHUNKS}
    params = _params("timeline", request) | {"corpus_chunks": len(corpus)}
    path = learning_cache.cache_path("timeline", topic or section or chapter or "", params)
    if use_cache and (cached := learning_cache.load(path)):
        return TimelineData.from_dict(cached), True
    if cache_only:                       # "library only" deployments never pay for a new timeline
        raise GenerationUnavailable("library")

    if topic:
        batches = [retrieve(topic, k=config.TIMELINE_CANDIDATES_K)]
    else:
        pool = sorted(corpus.scope_indices(chapter, section))
        batches = _batches(corpus, pool)

    def extract(sources: list[RetrievedChunk]):
        by_label = {s.source_id: s for s in sources}
        data = _json_with_one_retry(
            [{"role": "system", "content": EXTRACT_SYSTEM},
             {"role": "user", "content": _timeline_prompt(title, sources)}], gen)
        return validate_events(data.get("events") if isinstance(data, dict) else None, by_label)

    # Batches are independent, so run them concurrently (each call is slow).
    with ThreadPoolExecutor(max_workers=min(len(batches), 6)) as pool_:
        results = list(pool_.map(extract, batches))

    events: list[TimelineEvent] = []
    undated: list[TimelineEvent] = []
    omitted: list[str] = []
    for dated, loose, notes in results:
        events += dated
        undated += loose
        omitted += notes

    # merge duplicates that appear in different batches
    events, dupes = merge_duplicate_events(sorted(
        events, key=lambda e: (e.start_year, e.end_year or e.start_year)))
    undated, dupes2 = merge_duplicate_events(undated)
    timeline = TimelineData(title, events, undated, omitted + dupes + dupes2)
    learning_cache.save(path, params, timeline.to_dict())
    return timeline, False


