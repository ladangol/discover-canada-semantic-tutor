"""End-to-end latency of one spoken turn, measured stage by stage on the REAL pipeline (Milestone 4, section 1).

  python -m scripts.voice_eval.run_latency --warm 24          # repeated turns in one warm process
  python -m scripts.voice_eval.run_latency --cold 3           # N fresh processes: model load + first turn
  (the two are separate runs; each writes artifacts/voice_eval/latency_{warm,cold}.json)

A turn = synthetic speech (macOS `say` voice, NOT a person) -> Silero VAD + endpointer -> Whisper -> frozen retrieval -> LLM -> Kokoro.
Stages are timed with the production code paths (src/voice/live.py, session.py, tutor.py, tts.py); nothing in the app is changed.

What "latency" means here: from the last moment of speech to the first sample of the tutor's reply.
  endpoint  = AUDIO time the VAD needs to decide the learner has stopped (hangover, by design), measured on the sample clock
  asr, retrieval, generation = wall time inside each stage; tts = wall time to synthesise the WHOLE spoken reply (the app does not stream
  yet, so time-to-first-audio equals full synthesis; `tts_first_sentence` is what streaming WOULD give, reported separately as hypothetical)
  total     = endpoint + asr + retrieval + generation + tts (+ small glue). Browser/WebRTC transport and audio start-up are NOT included.
The real-time factor (synthesis seconds per second of audio) is reported on its own and is NOT a conversational latency.
Not measured: wake-word latency (Milestone 3 is not built).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np

from scripts.voice_eval import common as C


def _load_stack():
    """Load every model the way the app does, timing each load. Returns (corpus, timings)."""
    from src import tutor
    from src.corpus import load_corpus
    from src.embeddings import encode_query
    from src.voice import asr, tts
    from src.voice.vad import get_vad
    from src import config
    t = {}
    t0 = time.perf_counter(); vad = get_vad(config.live_settings()["vad"]); t["vad_load"] = time.perf_counter() - t0
    t0 = time.perf_counter(); asr.get_transcriber(); t["asr_load"] = time.perf_counter() - t0
    t0 = time.perf_counter(); corpus = load_corpus(); encode_query("warm up"); t["retrieval_load"] = time.perf_counter() - t0
    t0 = time.perf_counter(); tts.get_synthesizer(); t["tts_load"] = time.perf_counter() - t0
    return corpus, vad, t


def one_turn(q: dict, corpus, vad, voice: str, rng) -> dict:
    from src import tutor
    from src.voice import asr, audio, live, session as vsession, tts
    speech = C.say_samples(voice, q["text"])
    signal, end = C.stream(speech, noise_snr_db=None, rng=rng)
    engine = live.LiveEngine(vad=vad, cfg=live.endpoint_config())
    vad.reset()
    utt, at, vad_wall = C.capture(engine, signal)
    if utt is None:
        return {"id": q["id"], "voice": voice, "failed": "vad: no utterance emitted"}
    endpoint_s = (at - end) / C.SR
    vs = vsession.VoiceState()
    t0 = time.perf_counter()
    ok = live.ingest_utterance(vs, utt, asr.get_transcriber)
    asr_wall = time.perf_counter() - t0
    if not ok:
        return {"id": q["id"], "voice": voice, "failed": "asr: nothing recognised", "heard": vs.raw_heard}
    synth = tts.get_synthesizer()
    turn = vsession.submit_turn(vs, vs.heard, mode="ask", turn_fn=lambda t: tutor.handle_turn(t, corpus=corpus),
                                synth_fn=lambda t: synth.synthesize(tts.brief_speech(t, 350)))
    tm = turn.timings
    spoken = tts.brief_speech(turn.result.speech_text, 350)
    first = (tts.split_sentences(tts.prepare_speech_text(spoken)) or [spoken])[0]
    first_s = synth.synthesize(first).synthesis_seconds if turn.audio else None
    audio_s = audio.wav_seconds(turn.audio.audio_bytes) if turn.audio else None
    row = {"id": q["id"], "voice": voice, "heard": vs.raw_heard, "abstained": turn.result.abstained,
           "endpoint": endpoint_s, "vad_compute": vad_wall, "vad_ms_per_frame": 1000 * vad_wall / max(1, at // 512),
           "asr": tm.asr, "asr_wall": asr_wall, "retrieval": tm.retrieval, "generation": tm.generation, "tutor_total": tm.tutor_total,
           "tts": tm.tts, "tts_first_sentence": first_s, "speech_audio_s": len(trim_ := C.trim(speech)[0]) / C.SR,
           "reply_audio_s": audio_s, "asr_rtf": (tm.asr or 0) / max(1e-9, len(utt.samples) / C.SR),
           "tts_rtf": (tm.tts / audio_s) if (tm.tts and audio_s) else None}
    row["total"] = endpoint_s + (tm.asr or 0) + (tm.tutor_total or 0) + (tm.tts or 0)
    row["total_if_streamed"] = row["total"] - (tm.tts or 0) + (first_s or 0)
    return row


def one_turn_streamed(q: dict, corpus, vad, voice: str, rng, acks, max_chars: int = 240) -> dict:
    """The same turn as `one_turn`, through the Live page's streaming path: the LLM's text is spoken sentence by sentence while it is written
    (src/voice/speak.py), with a short acknowledgement pushed the moment the engine decides the learner has finished. The browser is replaced by a
    speaker that only records WHEN each piece of audio was handed over; WebRTC transport and playback start-up are not included."""
    from src import tutor
    from src.voice import asr, audio, live, session as vsession, speak, tts
    speech = C.say_samples(voice, q["text"])
    signal, end = C.stream(speech, noise_snr_db=None, rng=rng)
    pushes: list[tuple[float, float, str]] = []                 # (wall time, seconds of audio, kind)
    kind = ["ack"]
    speaker = speak.Speaker(lambda x: pushes.append((time.perf_counter(), len(x) / speak.SAMPLE_RATE, kind[0])))
    ack_pcm = acks.pick()
    ack_at: list[float] = []
    engine = live.LiveEngine(vad=vad, cfg=live.endpoint_config(),
                             on_utterance=lambda u: (speaker.push(ack_pcm), ack_at.append(time.perf_counter())) if ack_pcm is not None else None)
    vad.reset()
    t_cap0 = time.perf_counter()
    utt, at, vad_wall = C.capture(engine, signal)
    if utt is None:
        return {"id": q["id"], "voice": voice, "failed": "vad: no utterance emitted"}
    endpoint_s = (at - end) / C.SR
    vs = vsession.VoiceState()
    t0 = time.perf_counter()
    ok = live.ingest_utterance(vs, utt, asr.get_transcriber)
    asr_wall = time.perf_counter() - t0
    if not ok:
        return {"id": q["id"], "voice": voice, "failed": "asr: nothing recognised", "heard": vs.raw_heard}
    kind[0] = "reply"
    synth = tts.get_synthesizer()
    streamer = speak.SpeechStreamer(synth, speaker, max_chars=max_chars, verdict=tutor.speak_verdict)
    t_turn = time.perf_counter()
    turn = vsession.submit_turn(vs, vs.heard, mode="ask", turn_fn=lambda t: tutor.handle_turn(t, corpus=corpus, on_text=streamer.feed),
                                synth_fn=streamer.finish)
    t_done = time.perf_counter()
    tm = turn.timings
    reply = [p for p in pushes if p[2] == "reply"]
    first_after_asr = (streamer.first_audio_at - t_turn) if streamer.first_audio_at else None
    row = {"id": q["id"], "voice": voice, "heard": vs.raw_heard, "abstained": turn.result.abstained, "streamed": streamer.streamed,
           "endpoint": endpoint_s, "asr": tm.asr, "asr_wall": asr_wall, "retrieval": tm.retrieval, "generation": tm.generation,
           "tutor_total": tm.tutor_total, "turn_wall": t_done - t_turn, "reply_audio_s": sum(p[1] for p in reply),
           "first_audio_after_asr": first_after_asr, "ack_played": bool(ack_at), "ack_audio_s": sum(p[1] for p in pushes if p[2] == "ack")}
    row["total_to_first_audio"] = endpoint_s + (tm.asr or 0) + first_after_asr if first_after_asr is not None else None
    row["total_to_reply_done"] = endpoint_s + (tm.asr or 0) + (t_done - t_turn)
    row["sentences_pushed"] = len(reply)
    return row


def streamed(n: int) -> None:
    from src.voice import speak, tts
    corpus, vad, loads = _load_stack()
    acks = speak.Acks()
    acks.warm(tts.get_synthesizer())
    deadline = time.time() + 30
    while acks.pick() is None and time.time() < deadline:
        time.sleep(0.1)
    qs = [q for q in C.questions() if q.get("latency")]
    voices = ["samantha", "daniel", "karen", "rishi"]
    rng = np.random.default_rng(1)
    one_turn_streamed(qs[0], corpus, vad, "samantha", rng, acks)           # untimed warm-up
    rows = []
    for i in range(n):
        q, v = qs[i % len(qs)], voices[i % len(voices)]
        r = one_turn_streamed(q, corpus, vad, v, rng, acks)
        rows.append(r)
        print(f"[{i + 1}/{n}] {q['id']} {v:8} " + (f"first audio {r['total_to_first_audio']:.1f}s, done {r['total_to_reply_done']:.1f}s "
              f"(endpoint {r['endpoint']:.2f} asr {r['asr']:.2f} first-after-asr {r['first_audio_after_asr'] or float('nan'):.2f} sentences {r['sentences_pushed']})"
              if r.get("total_to_first_audio") else r.get("failed", "no reply audio")), flush=True)
    good = [r for r in rows if r.get("total_to_first_audio") is not None]
    keys = ["endpoint", "asr", "retrieval", "generation", "tutor_total", "first_audio_after_asr", "total_to_first_audio", "total_to_reply_done", "reply_audio_s"]
    summary = {k: C.pct([r[k] for r in good if r.get(k) is not None]) for k in keys}
    summary["answered_streamed"] = sum(bool(r["streamed"]) for r in good)
    summary["acknowledgement_played"] = sum(r["ack_played"] for r in rows)
    p = C.save("latency_streamed", {"kind": "streamed", "audio": "synthetic macOS `say` voices (US, UK, AU, IN accents), clean, 0.6 s lead + 2.5 s tail silence",
                                    "what": "Live page streaming path: LLM text spoken sentence by sentence, 240-character spoken budget, acknowledgement at the end of speech. "
                                            "Browser/WebRTC transport and playback start-up not included.",
                                    "runs": n, "ok": len(good), "summary": summary, "rows": rows})
    print("->", p)


def warm(n: int) -> None:
    corpus, vad, loads = _load_stack()
    qs = [q for q in C.questions() if q.get("latency")]
    voices = ["samantha", "daniel", "karen", "rishi"]
    rng = np.random.default_rng(1)
    one_turn(qs[0], corpus, vad, "samantha", rng)                  # untimed warm-up: first call of each model compiles/allocates
    rows = []
    for i in range(n):
        q, v = qs[i % len(qs)], voices[i % len(voices)]
        r = one_turn(q, corpus, vad, v, rng)
        rows.append(r)
        print(f"[{i + 1}/{n}] {q['id']} {v:8} " + (f"total {r['total']:.1f}s (endpoint {r['endpoint']:.2f} asr {r['asr']:.2f} tutor {r['tutor_total']:.2f} tts {r['tts']:.2f})"
                                                   if "total" in r else r["failed"]), flush=True)
    good = [r for r in rows if "total" in r]
    keys = ["endpoint", "vad_ms_per_frame", "asr", "retrieval", "generation", "tutor_total", "tts", "tts_first_sentence", "total", "total_if_streamed", "asr_rtf", "tts_rtf"]
    summary = {k: C.pct([r[k] for r in good if r.get(k) is not None]) for k in keys}
    summary["generation_when_answered"] = C.pct([r["generation"] for r in good if not r["abstained"] and r.get("generation") is not None])
    summary["generation_when_abstained"] = C.pct([r["generation"] for r in good if r["abstained"] and r.get("generation") is not None])
    p = C.save("latency_warm", {"kind": "warm", "audio": "synthetic macOS `say` voices (US, UK, AU, IN accents), clean, 0.6 s lead + 2.5 s tail silence",
                                "runs": n, "ok": len(good), "load_seconds_for_reference": loads, "summary": summary, "rows": rows})
    print("->", p)


def cold(n: int) -> None:
    rows = []
    for i in range(n):
        out = subprocess.run([sys.executable, "-m", "scripts.voice_eval.run_latency", "--cold-once", str(i)], capture_output=True, text=True,
                             cwd=str(C.ROOT))
        line = [l for l in out.stdout.splitlines() if l.startswith("COLD ")]
        if not line:
            print("cold run failed:", out.stderr[-300:])
            continue
        rows.append(json.loads(line[-1][5:]))
        r = rows[-1]
        print(f"[cold {i + 1}/{n}] load {sum(r['loads'].values()):.1f}s + first turn {r['turn']['total']:.1f}s" if "total" in r["turn"] else r["turn"], flush=True)
    good = [r for r in rows if "total" in r["turn"]]
    summary = {"model_load_total": C.pct([sum(r["loads"].values()) for r in good]), "first_turn_total": C.pct([r["turn"]["total"] for r in good]),
               "first_turn_asr": C.pct([r["turn"]["asr"] for r in good]), "first_turn_tts": C.pct([r["turn"]["tts"] for r in good]),
               "first_turn_retrieval": C.pct([r["turn"]["retrieval"] for r in good]),
               "first_turn_vs_load_plus_turn": C.pct([sum(r["loads"].values()) + r["turn"]["total"] for r in good]),
               **{f"load_{k}": C.pct([r["loads"][k] for r in good]) for k in ("vad_load", "asr_load", "retrieval_load", "tts_load")}}
    p = C.save("latency_cold", {"kind": "cold", "audio": "synthetic macOS `say` voice Samantha",
               "note": "A cold start = a fresh Python process (models loaded from disk, files likely in the OS page cache). A reboot/never-opened-model cold disk start was NOT measured.",
               "runs": n, "ok": len(good), "summary": summary, "rows": rows})
    print("->", p)


def cold_once(i: int) -> None:
    corpus, vad, loads = _load_stack()
    qs = [q for q in C.questions() if q.get("latency")]
    r = one_turn(qs[i % len(qs)], corpus, vad, "samantha", np.random.default_rng(1))
    print("COLD " + json.dumps({"loads": loads, "turn": r}, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warm", type=int, default=0)
    ap.add_argument("--cold", type=int, default=0)
    ap.add_argument("--stream", type=int, default=0, help="N turns through the streaming path (Live page), same questions as --warm")
    ap.add_argument("--cold-once", type=int, default=None)
    a = ap.parse_args()
    if a.cold_once is not None:
        cold_once(a.cold_once)
    elif a.stream:
        streamed(a.stream)
    elif a.warm:
        warm(a.warm)
    elif a.cold:
        cold(a.cold)
    else:
        ap.error("give --warm N, --stream N or --cold N")
