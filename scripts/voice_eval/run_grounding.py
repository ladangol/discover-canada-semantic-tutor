"""RAG grounding and answer quality of the full VOICE application (Milestone 4, section 5).

  python -m scripts.voice_eval.run_grounding

Each question is spoken (synthetic macOS voice) through VAD -> Whisper + name fixes -> the app's turn -> Kokoro. The frozen ChromaDB and
the Variant A top-5 retrieval are used as they are; the original held-out benchmark is NOT touched or re-run, and nothing is tuned against it.
Checks (all automatic, no LLM judge):
  abstain       a question the guide covers must be answered; one it does not cover must abstain
  expected      guide questions: the answer contains the facts listed in questions.json
  citation      every source shown was among the top-5 retrieved passages, and every "p. N" written in the answer is a shown source's page
  support       every year/number and every capitalised name in the answer also appears in the retrieved passages (a PROXY for "the evidence
                supports the answer": it catches invented dates/names, not a wrong paraphrase)
  asr->answer   if Whisper's text differs from the question, the typed question is also run and the two outcomes are compared
  spoken=shown  the text sent to the voice vs the text shown (word error rate between them) and a round trip: what Whisper hears when
                Kokoro reads the reply, vs the words Kokoro was given (numbers ignored, because spoken forms turn digits into words)
"""
from __future__ import annotations

import re

import jiwer
import numpy as np

from scripts.voice_eval import common as C
from scripts.voice_eval.run_tasks import Rig

NUM_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand first second third fourth fifth".split())
SKIP_CAPS = {"canada", "canadian", "canadians", "the", "a", "an", "it", "in", "he", "she", "they", "this", "that", "there", "his", "her", "i", "you", "yes", "no",
             "discover", "guide", "according", "for", "on", "at", "as", "by", "of", "to", "if", "when", "while", "after", "before", "during"}


def checkable(answer: str, said: str = "") -> tuple[set[str], set[str]]:
    """(numbers, capitalised words that are not sentence-initial) found in the answer, ignoring the citation brackets, contractions and words the
    learner said themselves (echoing a misheard name back is not inventing a fact)."""
    t = re.sub(r"\[[^\]]*\]", " ", answer)
    t = re.sub(r"[*_`#>]", "", t)
    t = re.sub(r"(?<=\w)['’]s\b", "", t)                       # "Tecumseh's" is the name Tecumseh
    t = re.sub(r"\b\w+['’](?:m|re|ve|ll|d|t)\b", " ", t)        # "I'm", "don't"
    spoken = set(C.norm(said).split())
    nums = set(re.findall(r"\b\d{3,4}\b|\b\d+(?:,\d{3})+\b", t))
    words = set()
    for sent in re.split(r"(?<=[.!?:])\s+|\n+", t):
        toks = re.findall(r"[A-Za-zÀ-ÿ'’-]+", sent)
        for w in toks[1:]:
            if w[:1].isupper() and C.norm(w) not in SKIP_CAPS and len(w) > 2 and C.norm(w) not in spoken:
                words.add(w)
    return nums, words


def drop_numbers(s: str) -> str:
    return " ".join(w for w in C.norm(s).split() if not w.isdigit() and w not in NUM_WORDS)


def run() -> dict:
    from src.retrieval import retrieve
    rig = Rig()
    tr = rig.asr.get_transcriber()
    synth = rig.tts.get_synthesizer()
    rows = []
    for q in [q for q in C.questions() if q["guide"] is not None]:
        heard, comp = rig.hear(q["text"])
        row = {"id": q["id"], "ref": q["text"], "guide": q["guide"], "heard": heard, "failed_at": comp}
        if heard is None:
            rows.append(row)
            continue
        turn = rig.send(heard)
        r = turn.result
        retrieved = retrieve(r.retrieval_query or heard, k=5)
        ids = {c.chunk_id for c in retrieved}
        evidence = C.compact(" ".join(c.text for c in retrieved))
        evidence_words = set(C.norm(" ".join(c.text for c in retrieved)).split())
        row.update(abstained=r.abstained, cited=[f"p.{c.page}" for c in r.cited], kind=r.kind, answer=r.text[:700])
        row["abstain_correct"] = (r.abstained != q["guide"])
        if q["guide"] and not r.abstained:
            row["expected_ok"] = all(C.compact(x) in C.compact(r.text) for x in q["expect"])
            pages_written = set(re.findall(r"\bp\.?\s?(\d+)", " ".join(re.findall(r"\[[^\]]*\]", r.text))))
            shown_pages = {p for c in r.cited for p in re.findall(r"\d+", c.page)}
            row["citation_ok"] = bool(r.cited) and all(c.chunk_id in ids for c in r.cited) and pages_written <= shown_pages
            nums, caps = checkable(r.text, heard)
            unsup_n = sorted(n for n in nums if n.replace(",", "") not in evidence.replace(",", ""))
            unsup_w = sorted(w for w in caps if C.norm(w).split()[0] not in evidence_words) if caps else []
            row.update(unsupported_numbers=unsup_n, unsupported_names=unsup_w, supported=not unsup_n and not unsup_w)
            row["shown_vs_spoken_wer"] = jiwer.wer(C.norm(re.sub(r"\[[^\]]*\]", " ", r.text)) or "x", C.norm(re.sub(r"\[[^\]]*\]", " ", r.speech_text)) or "x")
            spoken = rig.tts.brief_speech(r.speech_text, 350)
            from src.voice import audio as vaudio
            x, info = vaudio.read_wav(turn.audio.audio_bytes) if turn.audio else (None, None)
            if x is not None:
                keep, tr.initial_prompt = tr.initial_prompt, None          # hear it the way a listener without the hint would...
                try:
                    back = tr.transcribe(x, info.duration_seconds).text
                finally:
                    tr.initial_prompt = keep                                # ...and ALWAYS put the app's hint back for the next question
                a, b = drop_numbers(spoken), drop_numbers(back)
                row["roundtrip_wer_no_numbers"] = jiwer.wer(a, b) if a else None
        if C.norm(heard) != C.norm(q["text"]):
            typed = rig.tutor.handle_turn(q["text"], corpus=rig.corpus)
            row["asr_text_differs"] = True
            row["typed_abstained"] = typed.abstained
            row["same_decision"] = typed.abstained == r.abstained
            row["same_sources"] = {c.chunk_id for c in typed.cited} == {c.chunk_id for c in r.cited}
        rows.append(row)
        print(f"{q['id']} {'ans' if not r.abstained else 'abs'} cited={row['cited']} heard={heard!r}", flush=True)
    return {"rows": rows}


def rate(rows, key, cond=lambda r: True):
    g = [r for r in rows if key in r and cond(r)]
    return {"n": len(g), "ok": sum(bool(r[key]) for r in g)}


if __name__ == "__main__":
    out = run()
    rows = out["rows"]
    guide = lambda r: r["guide"] is True
    trap = lambda r: r["guide"] is False
    answered = [r for r in rows if guide(r) and r.get("abstained") is False]
    summary = {
        "questions": len(rows), "capture_failures": sum(r.get("heard") is None for r in rows),
        "guide_questions_answered": rate(rows, "abstain_correct", guide), "out_of_guide_abstained": rate(rows, "abstain_correct", trap),
        "answer_has_expected_facts": rate(answered, "expected_ok"), "citations_accurate": rate(answered, "citation_ok"),
        "answer_numbers_and_names_in_evidence": rate(answered, "supported"),
        "asr_text_differs": sum(bool(r.get("asr_text_differs")) for r in rows),
        "asr_error_changed_decision": sum(r.get("same_decision") is False for r in rows),
        "asr_error_changed_sources": sum(r.get("same_sources") is False for r in rows),
        "shown_vs_spoken_wer": C.pct([r["shown_vs_spoken_wer"] for r in answered if "shown_vs_spoken_wer" in r]),
        "roundtrip_wer_no_numbers": C.pct([r["roundtrip_wer_no_numbers"] for r in answered if r.get("roundtrip_wer_no_numbers") is not None]),
    }
    p = C.save("grounding", {"audio": "synthetic macOS `say` voice (Samantha), clean", "summary": summary, "rows": rows})
    for k, v in summary.items():
        print(f"{k:42} {v}")
    print("->", p)
