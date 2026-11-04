"""Turn artifacts/voice_eval/*.json into the markdown tables used in docs/VOICE_EVALUATION.md (so the report never contains a typed-in number).

  python -m scripts.voice_eval.summarize            # prints markdown and writes artifacts/voice_eval/summary.md
"""
from __future__ import annotations

from pathlib import Path

from scripts.voice_eval import common as C


def sec(x: float | None) -> str:
    return "–" if x is None else f"{x:.2f}"


def latency_tables() -> str:
    out = []
    w, c = C.load("latency_warm"), C.load("latency_cold")
    if w:
        s = w["summary"]
        out.append(f"**Warm** ({w['ok']}/{w['runs']} turns; {w['audio']}). Seconds.\n")
        out += ["| Stage | n | median | p90 | p95 | max |", "|---|---|---|---|---|---|"]
        names = [("endpoint", "VAD endpoint decision (audio time)"), ("asr", "Speech to text (Whisper)"), ("retrieval", "Retrieval (frozen top-5)"),
                 ("generation", "LLM generation"), ("tutor_total", "Tutor total (retrieval + LLM + glue)"), ("tts", "TTS, whole reply (= time to first audio today)"),
                 ("total", "**End of speech → first audio of the reply**"), ("tts_first_sentence", "(hypothetical) TTS of the first sentence only"),
                 ("total_if_streamed", "(hypothetical) total if speech were streamed per sentence")]
        for k, label in names:
            v = s[k]
            out.append(f"| {label} | {v['n']} | {sec(v['median'])} | {sec(v['p90'])} | {sec(v['p95'])} | {sec(v['max'])} |")
        out.append("")
        out.append("Not latencies (kept apart on purpose): ASR real-time factor median "
                   f"{s['asr_rtf']['median']:.2f}× (p95 {s['asr_rtf']['p95']:.2f}×); TTS real-time factor median {s['tts_rtf']['median']:.2f}× "
                   f"(p95 {s['tts_rtf']['p95']:.2f}×); VAD cost {s['vad_ms_per_frame']['median']:.2f} ms per 32 ms frame.\n")
    if c:
        s = c["summary"]
        out.append(f"**Cold** ({c['ok']}/{c['runs']} fresh processes). {c['note']} Seconds.\n")
        out += ["| Measure | n | median | p90 | p95 |", "|---|---|---|---|---|"]
        for k, label in (("load_vad_load", "Load VAD"), ("load_asr_load", "Load Whisper"), ("load_retrieval_load", "Load corpus + search model"), ("load_tts_load", "Load Kokoro"),
                         ("model_load_total", "All model loads"), ("first_turn_total", "First turn after loading (end of speech → first audio)"),
                         ("first_turn_vs_load_plus_turn", "Process start → first reply (loads + first turn)")):
            v = s[k]
            out.append(f"| {label} | {v['n']} | {sec(v['median'])} | {sec(v['p90'])} | {sec(v['p95'])} |")
        out.append("")
    return "\n".join(out)


def asr_tables() -> str:
    out = []
    for tag in ("asr_synthetic_small", "asr_synthetic_medium", "asr_real"):
        d = C.load(tag)
        if not d:
            continue
        out.append(f"**{tag}** — model `{d['model']}`, {d['clips']} clips per pipeline. Audio: {d['audio']}\n")
        out += ["| Pipeline | Audio | clips | WER | CER | exact transcripts | names wrong | clips with a name wrong | top-1 passage kept | name wrong **and** top-1 lost |", "|---|---|---|---|---|---|---|---|---|---|"]
        for k, v in d["scores"].items():
            pipe, cond = k.split(" | ")
            out.append(f"| {pipe} | {cond} | {v['clips']} | {v['wer']:.1%} | {v['cer']:.1%} | {v['exact_transcripts']:.0%} | {v['name_errors']}/{v['name_mentions']} "
                       f"| {v['clips_with_a_name_wrong']}/{v['clips_with_names']} | {v['retrieval_top1_kept']:.0%} | {v['name_wrong_and_retrieval_lost']} |")
        out.append("")
        out.append(f"Per name (misses/mentions, {'clean + noisy together' if any('SNR' in k for k in d['scores']) else 'clean only'}):\n")
        out += ["| Name | raw | hint | hint+fix |", "|---|---|---|---|"]
        for e, v in d["per_entity"].items():
            out.append(f"| {e} | {v['raw']} | {v['hint']} | {v['hint+fix']} |")
        out.append("")
        out.append("Per voice (clean, hint+fix): " + ", ".join(f"{k} {v['wer']:.1%}" for k, v in d["per_voice_clean_hint_fix"].items()) + "\n")
        out.append("Representative remaining errors (hint+fix):\n")
        out += ["| Said | Heard | Voice | SNR | name wrong | top-1 kept |", "|---|---|---|---|---|---|"]
        for f in d["failures"]:
            out.append(f"| {f['ref']} | {f['heard']} | {f['voice']} | {'clean' if f['snr_db'] is None else str(f['snr_db']) + ' dB'} | {', '.join(f['entities_missed']) or '–'} | {'yes' if f['ref_top1_kept'] else 'NO'} |")
        out.append("")
    return "\n".join(out)


def tasks_table() -> str:
    d = C.load("tasks")
    if not d:
        return ""
    out = [f"{d['audio']}. {d['steps']} steps: attempted {d['attempted']} → passed {d['passed']}, failed {d['failed']}; "
           f"not supported by the product {d['unsupported']}; not performed (needs Milestone 3) {d['not_performed']}.\n",
           "| Task | Said | Heard | Result | Component / detail |", "|---|---|---|---|---|"]
    for r in d["rows"]:
        out.append(f"| {r['task']} | {r['line'][:48]} | {r['heard'][:48]} | {r['status']} | {r['component']} {r['detail']} |".replace("\n", " "))
    out.append("\nFailures by component: " + (", ".join(f"{k} × {v}" for k, v in d["failures_by_component"].items()) or "none"))
    return "\n".join(out)


def grounding_table() -> str:
    d = C.load("grounding")
    if not d:
        return ""
    s = d["summary"]
    f = lambda v: f"{v['ok']}/{v['n']}"
    out = [f"{d['audio']}. {s['questions']} questions ({s['capture_failures']} not captured).\n", "| Check | Result |", "|---|---|",
           f"| Guide covers it → answered (not abstained) | {f(s['guide_questions_answered'])} |", f"| Guide does not cover it → abstained | {f(s['out_of_guide_abstained'])} |",
           f"| Answer contains the expected facts | {f(s['answer_has_expected_facts'])} |", f"| Citations accurate (shown sources ⊆ top-5; every written page is a shown source) | {f(s['citations_accurate'])} |",
           f"| Every year/number and name in the answer appears in the retrieved passages | {f(s['answer_numbers_and_names_in_evidence'])} |",
           f"| ASR text differed from the question | {s['asr_text_differs']} |", f"| …and that changed answer/abstain | {s['asr_error_changed_decision']} |",
           f"| …and that changed the cited sources | {s['asr_error_changed_sources']} |"]
    for k, label in (("shown_vs_spoken_wer", "Shown text vs text sent to the voice (WER)"), ("roundtrip_wer_no_numbers", "Round trip: Whisper hearing Kokoro vs words given (WER, numbers ignored)")):
        v = s[k]
        out.append(f"| {label} | n={v['n']}, median {v.get('median', 0):.1%}, p95 {v.get('p95', 0):.1%} |" if v.get("n") else f"| {label} | – |")
    bad = [r for r in d["rows"] if r.get("supported") is False or r.get("citation_ok") is False or r.get("expected_ok") is False or r.get("abstain_correct") is False]
    if bad:
        out += ["", "Representative failures:\n", "| Question | Heard | Problem |", "|---|---|---|"]
        for r in bad[:10]:
            why = []
            if r.get("abstain_correct") is False:
                why.append("answered an out-of-guide question" if not r["guide"] else "abstained on a guide question")
            if r.get("expected_ok") is False:
                why.append("missing expected fact")
            if r.get("citation_ok") is False:
                why.append("citation check failed")
            if r.get("supported") is False:
                why.append(f"not in evidence: {r.get('unsupported_numbers', []) + r.get('unsupported_names', [])}")
            out.append(f"| {r['ref']} | {r['heard']} | {'; '.join(why)} |")
    return "\n".join(out)


def key_numbers() -> dict[str, str]:
    """Every number quoted in the prose of the report comes from here (and so from a results file)."""
    import datetime
    k: dict[str, str] = {}
    w, c, t, g = C.load("latency_warm"), C.load("latency_cold"), C.load("tasks"), C.load("grounding")
    if w:
        s = w["summary"]
        k.update(warm_n=str(w["ok"]), warm_total_med=sec(s["total"]["median"]), warm_total_p90=sec(s["total"]["p90"]), warm_total_p95=sec(s["total"]["p95"]),
                 warm_total_min=sec(s["total"]["min"]), warm_total_max=sec(s["total"]["max"]),
                 warm_endpoint_med=sec(s["endpoint"]["median"]), warm_asr_med=sec(s["asr"]["median"]), warm_ret_med=sec(s["retrieval"]["median"]),
                 warm_gen_med=sec(s["generation"]["median"]), warm_gen_p95=sec(s["generation"]["p95"]), warm_tts_med=sec(s["tts"]["median"]),
                 warm_tts_p95=sec(s["tts"]["p95"]), warm_first_med=sec(s["tts_first_sentence"]["median"]), warm_streamed_med=sec(s["total_if_streamed"]["median"]),
                 warm_streamed_p95=sec(s["total_if_streamed"]["p95"]), tts_rtf_med=sec(s["tts_rtf"]["median"]), asr_rtf_med=sec(s["asr_rtf"]["median"]),
                 vad_ms_med=sec(s["vad_ms_per_frame"]["median"]), warm_tutor_med=sec(s["tutor_total"]["median"]), warm_tutor_p95=sec(s["tutor_total"]["p95"]),
                 warm_abstained=str(sum(bool(r.get("abstained")) for r in w["rows"] if "total" in r)))
    if c:
        s = c["summary"]
        k.update(cold_n=str(c["ok"]), cold_load_med=sec(s["model_load_total"]["median"]), cold_turn_med=sec(s["first_turn_total"]["median"]),
                 cold_all_med=sec(s["first_turn_vs_load_plus_turn"]["median"]))
    for tag, name in (("asr_synthetic_small", "small"), ("asr_synthetic_medium", "medium")):
        d = C.load(tag)
        if not d:
            continue
        sc = d["scores"]
        k[f"{name}_clips"] = str(d["clips"])
        for ent, ek in (("Michif", "michif"), ("Mi'kmaq", "mikmaq")):
            grp = [r for r in d["rows"] if r["pipeline"] == "hint+fix" and r["snr_db"] is None and ent in r["entities"]]      # clean only, so small and medium compare like with like
            k[f"{name}_{ek}"] = f"{sum(ent in r['entities_missed'] for r in grp)}/{len(grp)} wrong"
        for pipe, key in (("raw", "raw"), ("hint", "hint"), ("hint+fix", "fix")):
            for cond, ck in (("clean", "clean"), ("SNR 15 dB", "noisy")):
                v = sc.get(f"{pipe} | {cond}")
                if v:
                    k[f"{name}_{key}_wer_{ck}"] = f"{v['wer']:.1%}"
                    k[f"{name}_{key}_cer_{ck}"] = f"{v['cer']:.1%}"
                    k[f"{name}_{key}_names_{ck}"] = f"{v['name_errors']}/{v['name_mentions']}"
                    k[f"{name}_{key}_top1_{ck}"] = f"{v['retrieval_top1_kept']:.0%}"
                    k[f"{name}_{key}_lost_{ck}"] = f"{v['name_wrong_and_retrieval_lost']}/{v['clips']}"
    tm = C.load("asr_timing")
    if tm:
        for m, v in tm["models"].items():
            k[f"{m}_asr_time"] = sec(v["median"])
            k[f"{m}_asr_rtf"] = sec(v["rtf_median"])
        if {"small", "medium"} <= set(tm["models"]):
            k["medium_extra"] = sec(tm["models"]["medium"]["median"] - tm["models"]["small"]["median"])
    if t:
        k.update(tasks_attempted=str(t["attempted"]), tasks_passed=str(t["passed"]), tasks_failed=str(t["failed"]), tasks_unsupported=str(t["unsupported"]),
                 tasks_not_performed=str(t["not_performed"]))
    if g:
        s = g["summary"]
        f = lambda v: f"{v['ok']}/{v['n']}"
        k.update(g_questions=str(s["questions"]), g_answered=f(s["guide_questions_answered"]), g_abstained=f(s["out_of_guide_abstained"]),
                 g_expected=f(s["answer_has_expected_facts"]), g_citations=f(s["citations_accurate"]), g_support=f(s["answer_numbers_and_names_in_evidence"]),
                 g_asr_differs=str(s["asr_text_differs"]), g_asr_decision=str(s["asr_error_changed_decision"]), g_asr_sources=str(s["asr_error_changed_sources"]),
                 g_roundtrip_med=f"{s['roundtrip_wer_no_numbers']['median']:.1%}", g_roundtrip_p95=f"{s['roundtrip_wer_no_numbers']['p95']:.1%}")
    k["generated_on"] = datetime.date.today().isoformat()
    return k


def write_report() -> str:
    import json
    import re
    k = key_numbers()
    text = (Path(__file__).parent / "report_template.md").read_text()
    blocks = {"ENV": "```json\n" + json.dumps(C.environment(), indent=1) + "\n```", "LATENCY": latency_tables(), "ASR": asr_tables(), "TASKS": tasks_table(), "GROUNDING": grounding_table()}
    for name, body in blocks.items():
        text = text.replace("{{" + name + "}}", body)
    missing = sorted(set(re.findall(r"\[\[(\w+)\]\]", text)) - set(k))
    if missing:
        raise SystemExit(f"report_template.md uses numbers that no results file provides: {missing}")
    text = re.sub(r"\[\[(\w+)\]\]", lambda m: k[m.group(1)], text)
    (C.ROOT / "docs" / "VOICE_EVALUATION.md").write_text(text)
    return text


if __name__ == "__main__":
    import sys
    if "--write-report" in sys.argv:
        write_report()
        print("wrote docs/VOICE_EVALUATION.md")
        raise SystemExit(0)
    env = C.environment()
    md = ["## Environment\n", "```json", __import__("json").dumps(env, indent=1), "```\n", "## Latency\n", latency_tables(), "## ASR\n", asr_tables(), "## Tasks\n", tasks_table(),
          "\n## Grounding\n", grounding_table()]
    text = "\n".join(md)
    (C.OUT / "summary.md").write_text(text)
    print(text)
