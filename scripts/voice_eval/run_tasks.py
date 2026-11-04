"""Conversational task success (Milestone 4, section 4): scripted spoken interactions through the REAL pipeline.

  python -m scripts.voice_eval.run_tasks

Each step speaks a line (synthetic macOS voice) -> Silero VAD + endpointer -> Whisper + name fixes -> the app's turn logic (src/voice/session.py,
src/tutor.py, real LLM). A step passes only if its checks pass; a failure is assigned to the FIRST component that broke:
vad/endpoint -> asr -> option-matching/intent -> retrieval -> generation/grounding -> state. Steps the product does not support are
reported as "not supported" (they are NOT silently counted as passes or hidden), and steps that need Milestone 3 (interruption) as "not performed".
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from scripts.voice_eval import common as C

VOICE = "samantha"


@dataclass
class Step:
    task: str
    line: str
    status: str = "pass"             # pass | fail | unsupported | not performed
    component: str = ""
    detail: str = ""
    heard: str = ""
    extra: dict = field(default_factory=dict)


class Rig:
    def __init__(self):
        from src import tutor
        from src.corpus import load_corpus
        from src.embeddings import encode_query
        from src.voice import asr, live, session as vsession, tts
        from src.voice.vad import get_vad
        self.tutor, self.asr, self.live, self.vsession, self.tts = tutor, asr, live, vsession, tts
        self.corpus = load_corpus()
        encode_query("warm up")
        self.vad = get_vad("silero")
        self.vs = vsession.VoiceState()
        self.rng = np.random.default_rng(5)

    def hear(self, line: str, snr: float | None = None) -> tuple[str | None, str]:
        """Speak `line` into the live engine and transcribe what it endpointed. Returns (draft text or None, failing component)."""
        sig, _ = C.stream(C.say_samples(VOICE, line), noise_snr_db=snr, rng=self.rng)
        self.vad.reset()
        utt, _, _ = C.capture(self.live.LiveEngine(vad=self.vad, cfg=self.live.endpoint_config()), sig)
        if utt is None:
            return None, "vad/endpoint"
        if not self.live.ingest_utterance(self.vs, utt, self.asr.get_transcriber):
            return None, "asr"
        return self.vs.heard, ""

    def send(self, text: str, quiz=None):
        synth = self.tts.get_synthesizer()
        return self.vsession.submit_turn(self.vs, text, mode="quiz" if quiz else "ask",
                                         turn_fn=lambda t: self.tutor.handle_turn(t, quiz=quiz, corpus=self.corpus),
                                         synth_fn=lambda t: synth.synthesize(self.tts.brief_speech(t, 350)))


def _ask(rig: Rig, task: str, line: str, *, expect: list[str], must_answer: bool, entities: list[str]) -> Step:
    s = Step(task, line)
    heard, comp = rig.hear(line)
    if heard is None:
        s.status, s.component, s.detail = "fail", comp, "nothing usable was captured"
        return s
    s.heard = heard
    if [e for e in entities if not C.entity_found(e, heard)]:
        s.status, s.component, s.detail = "fail", "asr", f"name misheard: {[e for e in entities if not C.entity_found(e, heard)]}"
        return s
    turn = rig.send(heard)
    r = turn.result
    s.extra = {"abstained": r.abstained, "cited": [f"p.{c.page}" for c in r.cited], "audio": turn.audio is not None}
    if must_answer:
        if r.abstained:
            s.status, s.component, s.detail = "fail", "retrieval", "abstained although the guide answers it"
        elif not all(C.compact(x) in C.compact(r.text) for x in expect):
            s.status, s.component, s.detail = "fail", "generation/grounding", f"answer lacks {expect}"
        elif not r.cited:
            s.status, s.component, s.detail = "fail", "generation/grounding", "no citation shown"
        elif turn.audio is None:
            s.status, s.component, s.detail = "fail", "tts", turn.audio_note or "no audio"
    else:
        if not r.abstained:
            s.status, s.component, s.detail = "fail", "generation/grounding", "answered although the guide does not cover it"
    return s


def _quiz_answers(rig: Rig, n_quizzes: int = 3) -> list[Step]:
    steps = []
    ordinals = ["first", "second", "third", "fourth"]
    for j in range(n_quizzes):
        try:
            q = rig.tutor.generate_quiz(rig.corpus, kind="mcq", rng=random.Random(100 + j))
        except Exception as exc:
            steps.append(Step("quiz: generate a question", "(button)", "fail", "generation/grounding", type(exc).__name__))
            continue
        forms = [("letter", lambda k, q=q: f"Option {'ABCD'[k]}."), ("ordinal", lambda k, q=q: f"The {ordinals[k]} one."),
                 ("answer text", lambda k, q=q: q.options[k])]
        for f, (form, say) in enumerate(forms):
            k = (j + f) % 4
            line = say(k)
            s = Step(f"quiz answer by {form}", line)
            heard, comp = rig.hear(line)
            if heard is None:
                s.status, s.component, s.detail = "fail", comp, "nothing usable was captured"
                steps.append(s)
                continue
            s.heard = heard
            turn = rig.send(heard, quiz=q)
            r = turn.result
            if r.kind != "quiz_feedback":
                # was the transcript at least right? then it is the option matcher, otherwise ASR
                right = C.compact(line) in C.compact(heard) or C.compact(heard) in C.compact(line)
                s.status, s.component = "fail", ("intent/option matching" if right else "asr")
                s.detail = f"not graded ({r.kind}); heard {heard!r}"
            elif r.chosen_index != k:
                s.status, s.component, s.detail = "fail", "asr" if C.compact(line) not in C.compact(heard) else "intent/option matching", \
                    f"graded option {r.chosen_index}, meant {k}; heard {heard!r}"
            elif r.quiz_result.correct != (k == q.correct_index):
                s.status, s.component, s.detail = "fail", "state", "grade does not match the chosen option"
            steps.append(s)
    return steps


def _mumble(rig: Rig) -> Step:
    q = rig.tutor.generate_quiz(rig.corpus, kind="mcq", rng=random.Random(777))
    s = Step("quiz: mumbled non-answer is not graded", "Hmm, I don't know.")
    heard, comp = rig.hear(s.line)
    if heard is None:
        s.status, s.component, s.detail = "fail", comp, "nothing captured"
        return s
    s.heard = heard
    r = rig.send(heard, quiz=q).result
    if r.kind != "clarify" or r.quiz_result is not None:
        s.status, s.component, s.detail = "fail", "intent/option matching", f"was treated as {r.kind}"
    return s


def _edit_transcript(rig: Rig) -> Step:
    s = Step("correct the transcript before sending", "What did Laura Secord do in 1813?")
    heard, comp = rig.hear(s.line)
    if heard is None:
        s.status, s.component, s.detail = "fail", comp, "nothing captured"
        return s
    raw = rig.vs.raw_heard
    edited = "Who was Laura Secord?"
    n_before = len(rig.vs.turns)
    turn = rig.send(edited)
    problems = [m for ok, m in ((turn.edited, "edit not flagged"), (turn.heard == raw, "original ASR text lost"), (turn.said == edited, "sent text differs from the edit"),
                                (not rig.vs.has_draft, "draft not cleared"), (len(rig.vs.turns) == n_before + 1, "turn count wrong")) if not ok]
    s.heard = raw
    if problems:
        s.status, s.component, s.detail = "fail", "state", "; ".join(problems)
    return s


def run() -> list[Step]:
    rig = Rig()
    steps: list[Step] = []
    steps.append(_ask(rig, "ask a question, get a grounded spoken answer", "What does RCMP stand for?", expect=["Royal Canadian Mounted Police"], must_answer=True, entities=["RCMP"]))
    steps.append(_ask(rig, "ask a question with a name", "What did Chief Tecumseh do in the War of 1812?", expect=["Brock"], must_answer=True, entities=["Tecumseh"]))
    steps.append(_ask(rig, "ask a question with a name", "Who was Samuel de Champlain?", expect=["1608"], must_answer=True, entities=["Samuel de Champlain"]))
    steps.append(_ask(rig, "ask something the guide does not cover (must abstain)", "Who was Louis-Joseph Papineau?", expect=[], must_answer=False, entities=["Louis-Joseph Papineau"]))
    steps.append(_ask(rig, "ask something the guide does not cover (must abstain)", "Who is the current prime minister of Canada?", expect=[], must_answer=False, entities=[]))
    steps.extend(_quiz_answers(rig))
    steps.append(_mumble(rig))
    steps.append(_edit_transcript(rig))

    # --- things the product does not do today: measured, not hidden ------------------------------------------------------------------
    q = Step("request a quiz by voice", "Quiz me about the RCMP.")
    heard, comp = rig.hear(q.line)
    if heard is None:
        q.status, q.component, q.detail = "fail", comp, "nothing captured"
    else:
        q.heard = heard
        r = rig.send(heard).result
        q.status, q.component = "unsupported", "intent (voice commands are not implemented; the quiz starts from a button)"
        q.detail = f"treated as an ordinary question → {'abstained' if r.abstained else 'answered'}"
    steps.append(q)

    f = Step("ask for clarification of the previous answer", "Can you explain that again in simpler words?")
    heard, comp = rig.hear(f.line)
    if heard is None:
        f.status, f.component, f.detail = "fail", comp, "nothing captured"
    else:
        f.heard = heard
        r = rig.send(heard).result
        f.status, f.component = "unsupported", "dialogue memory (each turn is answered on its own; 'that' has no referent)"
        f.detail = f"outcome: {'abstained (honest)' if r.abstained else 'answered with: ' + r.text[:90]}"
    steps.append(f)

    for t, line in (("interrupt an explanation", "(speak over the tutor)"), ("continue studying after an interruption", "(resume the same topic)")):
        steps.append(Step(t, line, "not performed", "barge-in", "Milestone 3 (barge-in) is not built; there is nothing to interrupt yet"))

    consistent = [t.id for t in rig.vs.turns] == sorted(t.id for t in rig.vs.turns) and not rig.vs.has_draft
    steps.append(Step("dialogue state stays consistent over the whole session", f"{len(rig.vs.turns)} turns", "pass" if consistent else "fail",
                      "" if consistent else "state", "ids increase, no draft left over" if consistent else "turn ids or draft inconsistent"))
    return steps


if __name__ == "__main__":
    steps = run()
    done = [s for s in steps if s.status in ("pass", "fail")]
    by = {}
    for s in steps:
        by.setdefault(s.status, []).append(s)
    for s in steps:
        print(f"{s.status:13} {s.task:55} {s.line[:40]!r:44} {s.component} {s.detail}")
    fail_components: dict[str, int] = {}
    for s in by.get("fail", []):
        fail_components[s.component] = fail_components.get(s.component, 0) + 1
    p = C.save("tasks", {"audio": f"synthetic macOS `say` voice ({VOICE}), clean", "steps": len(steps), "attempted": len(done),
                         "passed": len(by.get("pass", [])), "failed": len(by.get("fail", [])), "unsupported": len(by.get("unsupported", [])),
                         "not_performed": len(by.get("not performed", [])), "failures_by_component": fail_components,
                         "rows": [s.__dict__ for s in steps]})
    print(f"\nattempted {len(done)}: pass {len(by.get('pass', []))}, fail {len(by.get('fail', []))} · unsupported {len(by.get('unsupported', []))} · not performed {len(by.get('not performed', []))}")
    print("->", p)
