"""Development tool: how well does the ASR hear Canadian names from the guide?

Two modes (neither runs in the app, and neither is a benchmark until real recordings are collected):

  python scripts/voice_proper_nouns.py --voice /path/voice.onnx
      SYNTHETIC smoke check: Piper reads each phrase, Whisper transcribes it, jiwer scores it. Synthetic speech is easier
      than a person's voice, so treat the numbers as a sanity check only — do not quote them as accuracy.

  python scripts/voice_proper_nouns.py --recordings DIR
      REAL speech: DIR holds *.wav files plus references.tsv (one "filename<TAB>reference text" per line).
      Only this mode yields numbers you can report.

Nothing is uploaded; no key is used. Requires requirements-voice.txt.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PHRASES = ["Sir John A. Macdonald", "Louis-Hippolyte La Fontaine", "Laura Secord", "James FitzGibbon", "Tecumseh",
           "Nunavut", "Iqaluit", "Confederation", "Who was Sir John A. Macdonald?", "Name two of the original provinces."]


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower().replace("-", " "))).strip()


def synthesize(voice_path: Path, text: str) -> bytes:
    from piper import PiperVoice
    v = PiperVoice.load(str(voice_path))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        v.synthesize_wav(text, w)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", type=Path, help="Piper .onnx voice (synthetic smoke check)")
    ap.add_argument("--recordings", type=Path, help="folder with *.wav and references.tsv (real speech)")
    ap.add_argument("--out", type=Path, help="optional JSON file for the per-phrase results")
    a = ap.parse_args()
    if bool(a.voice) == bool(a.recordings):
        ap.error("give exactly one of --voice or --recordings")
    import jiwer
    from src.voice import asr, audio

    if a.recordings:
        refs = dict(line.rstrip("\n").split("\t", 1) for line in (a.recordings / "references.tsv").read_text().splitlines() if line.strip())
        items = [(name, ref, (a.recordings / name).read_bytes()) for name, ref in refs.items()]
        kind = "REAL recordings"
    else:
        items = [(p, p, synthesize(a.voice, p)) for p in PHRASES]
        kind = "SYNTHETIC speech (smoke check, not accuracy)"

    tr = asr.get_transcriber()
    rows, refs_l, hyps_l = [], [], []
    for label, ref, wav in items:
        x, info = audio.read_wav(wav)
        res = tr.transcribe(x, info.duration_seconds)
        refs_l.append(norm(ref)), hyps_l.append(norm(res.text))
        rows.append({"item": label, "reference": ref, "heard": res.text, "exact": norm(ref) == norm(res.text),
                     "audio_s": round(info.duration_seconds, 2), "asr_s": round(res.inference_seconds, 2)})
        print(f"{'ok ' if rows[-1]['exact'] else 'DIFF'} {ref!r:45} -> {res.text!r}   ({res.inference_seconds:.2f}s for {info.duration_seconds:.1f}s)")
    wer, cer = jiwer.wer(refs_l, hyps_l), jiwer.cer(refs_l, hyps_l)
    print(f"\n{kind}: {len(rows)} items · WER {wer:.1%} · CER {cer:.1%}  (normalized: lower-case, punctuation removed)")
    if a.out:
        a.out.write_text(json.dumps({"kind": kind, "wer": wer, "cer": cer, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
