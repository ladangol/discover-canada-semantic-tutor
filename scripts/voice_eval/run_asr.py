"""Speech-recognition accuracy on Discover Canada questions (Milestone 4, section 2).

  python -m scripts.voice_eval.run_asr [--model small] [--voices samantha,daniel,karen,rishi,kokoro] [--snr 15]
  python -m scripts.voice_eval.run_asr --recordings DIR       # REAL recordings: DIR/*.wav + DIR/references.tsv ("file<TAB>reference")

Three pipelines are compared on identical audio:
  raw        Whisper with no hint, no correction
  hint       Whisper with the app's vocabulary hint (config.ASR_INITIAL_PROMPT_DEFAULT)
  hint+fix   hint, then the app's name corrector (src/voice/names.correct_names)  <- what the app really does
Scores (jiwer, lower-cased, accent-folded, punctuation removed): WER and CER overall; name errors per mention (a name counts as heard only
if it appears, ignoring spaces/hyphens/accents/case); and whether a mistake could change the question's MEANING, which is measured two
objective ways (no human judgement): the name is wrong AND/OR the frozen top-5 retrieval for the heard text no longer contains the passage
that the correctly spelled question retrieved first.

The audio is SYNTHETIC (macOS voices + the app's own Kokoro voice), clean or with pink noise at the given SNR. Synthetic speech is easier
than a person's, so these numbers are a lower bound on the real error rate and must be reported as such.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from scripts.voice_eval import common as C

PIPELINES = ("raw", "hint", "hint+fix")


def _retrieve_ids(text: str) -> list[str]:
    from src import tutor
    from src.retrieval import retrieve
    return [c.chunk_id for c in retrieve(tutor.search_text(text), k=5)]


def synthetic(models: list[str], voices: list[str], snrs: list[float | None], limit: int) -> list[dict]:
    from src import config
    from src.voice import asr, names
    s = config.asr_settings()
    tr = asr.FasterWhisperTranscriber(models[0], s["device"], s["compute_type"], s["language"], None)
    hint_prompt = config.ASR_INITIAL_PROMPT_DEFAULT
    qs = C.questions()[:limit] if limit else C.questions()
    rng = np.random.default_rng(3)
    ref_cache: dict[str, list[str]] = {}
    rows = []
    n = len(qs) * len(voices) * len(snrs)
    k = 0
    for q in qs:
        ref_cache[q["id"]] = _retrieve_ids(q["text"])
        for v in voices:
            clean = C.speech_samples(v, q["text"])
            for snr in snrs:
                k += 1
                x = clean if snr is None else C.add_noise(clean, snr, rng)
                heard = {}
                tr.initial_prompt = None
                r = tr.transcribe(x, len(x) / C.SR)
                heard["raw"], t_raw = r.text, r.inference_seconds
                tr.initial_prompt = hint_prompt
                r = tr.transcribe(x, len(x) / C.SR)
                heard["hint"], t_hint = r.text, r.inference_seconds
                heard["hint+fix"] = names.correct_names(heard["hint"])[0]
                for p in PIPELINES:
                    ids = _retrieve_ids(heard[p])
                    ref_ids = ref_cache[q["id"]]
                    rows.append({"id": q["id"], "voice": v, "snr_db": snr, "pipeline": p, "ref": q["text"], "heard": heard[p],
                                 "entities": q["entities"], "entities_missed": [e for e in q["entities"] if not C.entity_found(e, heard[p])],
                                 "audio_s": len(x) / C.SR, "asr_s": t_raw if p == "raw" else t_hint,
                                 "ref_top1_kept": ref_ids[0] in ids, "top1_same": ids[0] == ref_ids[0], "overlap5": len(set(ids) & set(ref_ids)) / 5})
        if k % 10 == 0 or k == n:
            print(f"  {k}/{n} clips", flush=True)
    return rows


def real(dir_) -> list[dict]:
    """Real recordings: scored with the same three pipelines."""
    from src import config
    from src.voice import asr, audio, names
    refs = dict(l.split("\t", 1) for l in (dir_ / "references.tsv").read_text().splitlines() if l.strip())
    byref = {q["text"].lower(): q for q in C.questions()}
    s = config.asr_settings()
    tr = asr.FasterWhisperTranscriber(s["model"], s["device"], s["compute_type"], s["language"], None)
    rows = []
    for name, ref in refs.items():
        x, info = audio.read_wav((dir_ / name).read_bytes())
        ents = byref.get(ref.strip().lower(), {}).get("entities", [])
        ref_ids = _retrieve_ids(ref)
        heard = {}
        tr.initial_prompt = None
        heard["raw"] = tr.transcribe(x, info.duration_seconds).text
        tr.initial_prompt = config.ASR_INITIAL_PROMPT_DEFAULT
        heard["hint"] = tr.transcribe(x, info.duration_seconds).text
        heard["hint+fix"] = names.correct_names(heard["hint"])[0]
        for p in PIPELINES:
            ids = _retrieve_ids(heard[p])
            rows.append({"id": name, "voice": "real", "snr_db": None, "pipeline": p, "ref": ref, "heard": heard[p], "entities": ents,
                         "entities_missed": [e for e in ents if not C.entity_found(e, heard[p])], "audio_s": info.duration_seconds, "asr_s": None,
                         "ref_top1_kept": ref_ids[0] in ids, "top1_same": ids[0] == ref_ids[0], "overlap5": len(set(ids) & set(ref_ids)) / 5})
    return rows


def score(rows: list[dict]) -> dict:
    import jiwer
    out = {}
    groups = sorted({(r["pipeline"], r["snr_db"]) for r in rows}, key=lambda g: (g[1] is not None, g[1] or 0, PIPELINES.index(g[0])))
    for pipe, snr in groups:
        g = [r for r in rows if r["pipeline"] == pipe and r["snr_db"] == snr]
        ref, hyp = [C.norm(r["ref"]) for r in g], [C.norm(r["heard"]) for r in g]
        mentions = sum(len(r["entities"]) for r in g)
        missed = sum(len(r["entities_missed"]) for r in g)
        with_ent = [r for r in g if r["entities"]]
        key = f"{pipe} | {'clean' if snr is None else f'SNR {snr:g} dB'}"
        out[key] = {"clips": len(g), "wer": jiwer.wer(ref, hyp), "cer": jiwer.cer(ref, hyp), "name_mentions": mentions, "name_errors": missed,
                    "name_error_rate": missed / mentions if mentions else None,
                    "exact_transcripts": sum(a == b for a, b in zip(ref, hyp)) / len(g),
                    "clips_with_a_name_wrong": sum(bool(r["entities_missed"]) for r in with_ent), "clips_with_names": len(with_ent),
                    "retrieval_top1_kept": sum(r["ref_top1_kept"] for r in g) / len(g),
                    "retrieval_top1_same": sum(r["top1_same"] for r in g) / len(g),
                    "name_wrong_and_retrieval_lost": sum(bool(r["entities_missed"]) and not r["ref_top1_kept"] for r in g)}
    return out


def per_entity(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        for e in r["entities"]:
            d = out.setdefault(e, {p: [0, 0] for p in PIPELINES})
            d[r["pipeline"]][1] += 1
            d[r["pipeline"]][0] += e in r["entities_missed"]
    return {e: {p: f"{m}/{n}" for p, (m, n) in d.items()} for e, d in sorted(out.items())}


def per_voice(rows: list[dict]) -> dict:
    import jiwer
    out = {}
    for v in sorted({r["voice"] for r in rows}):
        g = [r for r in rows if r["voice"] == v and r["pipeline"] == "hint+fix" and r["snr_db"] is None]
        if g:
            out[v] = {"clips": len(g), "wer": jiwer.wer([C.norm(r["ref"]) for r in g], [C.norm(r["heard"]) for r in g])}
    return out


def failures(rows: list[dict], limit: int = 14) -> list[dict]:
    bad = [r for r in rows if r["pipeline"] == "hint+fix" and C.norm(r["ref"]) != C.norm(r["heard"])]
    bad.sort(key=lambda r: (not r["entities_missed"], r["ref_top1_kept"]))
    return [{k: r[k] for k in ("ref", "heard", "voice", "snr_db", "entities_missed", "ref_top1_kept")} for r in bad[:limit]]


def timing(models: list[str], n: int = 12) -> dict:
    """Whisper speed per model size on the SAME clips, run alone (accuracy runs may overlap with other work, so their timings are not used)."""
    from src import config
    from src.voice import asr
    s = config.asr_settings()
    qs = [q for q in C.questions() if q.get("latency")][:n]
    clips = [C.say_samples("samantha", q["text"]) for q in qs]
    out = {}
    for m in models:
        tr = asr.FasterWhisperTranscriber(m, s["device"], s["compute_type"], s["language"], config.ASR_INITIAL_PROMPT_DEFAULT)
        tr.transcribe(clips[0], len(clips[0]) / C.SR)                               # warm-up
        secs = [tr.transcribe(x, len(x) / C.SR).inference_seconds for x in clips]
        out[m] = {**C.pct(secs), "rtf_median": float(np.median([t / (len(x) / C.SR) for t, x in zip(secs, clips)]))}
        print(f"{m}: median {out[m]['median']:.2f}s  RTF {out[m]['rtf_median']:.2f}")
    return out


if __name__ == "__main__":
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="Whisper size for the synthetic run (default: the app's)")
    ap.add_argument("--voices", default="kokoro,samantha,daniel,karen,rishi")
    ap.add_argument("--snr", default="15", help="comma list of noise SNRs in dB (clean is always run); '' = clean only")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--recordings", type=Path)
    ap.add_argument("--timing", default="", help="comma list of model sizes: only time them on the same 12 clips (no accuracy run)")
    a = ap.parse_args()
    if a.timing:
        print("->", C.save("asr_timing", {"audio": "synthetic macOS `say` voice (Samantha), same 12 clips, vocabulary hint on, run alone", "models": timing(a.timing.split(","))}))
        raise SystemExit(0)
    from src import config
    model = a.model or config.asr_settings()["model"]
    if a.recordings:
        rows, tag = real(a.recordings), "asr_real"
        audio_note = f"REAL recordings from {a.recordings.name}"
    else:
        snrs = [None] + [float(s) for s in a.snr.split(",") if s]
        t0 = time.perf_counter()
        rows, tag = synthetic([model], a.voices.split(","), snrs, a.limit), f"asr_synthetic_{model}"
        audio_note = "SYNTHETIC speech (macOS `say` voices + the app's Kokoro voice); pink noise where an SNR is given. Not people."
        print(f"took {time.perf_counter() - t0:.0f}s")
    scores = score(rows)
    p = C.save(tag, {"model": model, "audio": audio_note, "clips": len(rows) // len(PIPELINES), "scores": scores, "per_entity": per_entity(rows),
                     "per_voice_clean_hint_fix": per_voice(rows), "failures": failures(rows), "rows": rows})
    for k, v in scores.items():
        print(f"{k:28} WER {v['wer']:.1%}  CER {v['cer']:.1%}  names wrong {v['name_errors']}/{v['name_mentions']}  top-1 kept {v['retrieval_top1_kept']:.0%}")
    print("->", p)
