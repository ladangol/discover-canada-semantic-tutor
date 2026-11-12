"""Spoken commands for the Live page, recognised by simple rules (no model, no network, no tutor/retrieval imports).

The learner never has to click to start or leave a quiz question: they say so.

    "ask me a multiple choice question" / "quiz me" / "give me an open question"  -> a new question (kind: "mcq" or "free")
    "another question" / "next one" / "skip"                                       -> a new question of the same kind as the last
    "stop the quiz" / "never mind" / "back to questions"                           -> leave the open question

Only SHORT utterances can be commands (a real question about the guide is never mistaken for one), and each rule needs the word "me" or "question"/"quiz"
in a command-like phrase. Anything else is left alone: while a question is open it is an answer, otherwise it is a question for the guide."""
from __future__ import annotations

import re

MAX_COMMAND_WORDS = 12


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' -]", " ", (text or "").lower().replace("’", "'")).replace("-", " ").split()


def _kind(words: list[str], last_kind: str | None) -> str:
    joined = " ".join(words)
    if re.search(r"\b(multiple choice|mcq|options?|choices?)\b", joined):
        return "mcq"
    if re.search(r"\b(open|free|written|short answer|free response|open ended)\b", joined):
        return "free"
    return last_kind or "mcq"


def quiz_request(text: str, last_kind: str | None = None) -> str | None:
    """'mcq' / 'free' if the learner asked to be quizzed, else None."""
    w = _words(text)
    if not w or len(w) > MAX_COMMAND_WORDS:
        return None
    j = " ".join(w)
    asks_me = re.search(r"\b(ask|give|get|throw|set|pick)\s+me\b", j) and re.search(r"\b(question|quiz|one|another|something)\b", j)
    quiz_me = re.search(r"\b(quiz|test)\s+me\b|\btest my knowledge\b|\bpractice (quiz|question)s?\b|\bstart (a |the )?quiz\b", j)
    musing = re.search(r"\b(my|think|thinking|thought|about|gonna|going|wondering)\b", j)                      # "I'm thinking about my next question" is not a request
    again = not musing and re.search(r"\b(another|next|new|different|one more)\b.*\b(question|one)\b|\bnext question\b|^skip( this| it| that)?( one| question)?$|^(i )?skip$", j)
    typed = re.search(r"\b(multiple choice|mcq|open ended|open|free response) question\b", j) and len(w) <= 8 and \
        not re.search(r"\b(what|who|when|where|why|how|which)\b", j)
    if asks_me or quiz_me or again or typed:
        return _kind(w, last_kind)
    return None


def quiz_exit(text: str) -> bool:
    """True if the learner wants to leave the open quiz question without answering."""
    w = _words(text)
    if not w or len(w) > MAX_COMMAND_WORDS:
        return False
    j = " ".join(w)
    return bool(re.search(r"\b(stop|end|exit|quit|cancel|leave)\b.*\b(quiz|question|practice)\b|\bnever ?mind\b|\bforget (it|that)\b|\bback to (the )?(normal|questions?|asking)\b|\bno more (questions?|quiz)\b", j))


# Sounds that are not a message: hesitations (however long the learner stretches them), breaths picked up as speech, reactions to the answer ("cool",
# "nice"), and the words Whisper tends to invent on near-silence.
FILLERS = {"um", "umm", "uh", "uhh", "uh huh", "hmm", "hm", "hmmm", "mm", "mmm", "mhm", "mm hmm", "ah", "aah", "oh", "er", "erm", "huh", "ha", "hah", "eh",
           "you", "bye", "thank you", "thanks", "okay", "ok", "so", "and", "the", "i", "yeah", "yes", "cool", "nice", "great", "right", "well", "like", "alright",
           "all right", "wow", "ooh", "aha", "interesting", "let me think", "let's see", "lets see", "hold on", "wait", "one sec", "one second", "sure", "good", "awesome"}
_HESITATION = re.compile(r"^(?:u+h*m+|u+h+|h+m+|m+h*m*|e+r+m*|e+h+|a+h+|o+h+|h+u+h+|u+m+h+)$")
# "yes"/"okay" are fillers only when nothing is waiting for them: a pending "Did you mean…?" or an open question is handled before this check (see is_noise).


# Words that carry no request on their own ("let me think about it", "give me a second"): they only count as noise when EVERY word of the utterance is
# a filler or one of these, so "let me think about who Papineau was" is still a question.
SOFT = {"let", "me", "think", "thinking", "about", "it", "that", "this", "for", "a", "an", "moment", "second", "sec", "bit", "just", "give", "minute", "i'm",
        "im", "am", "my", "next", "question", "thought", "gonna", "going", "to", "hang", "on"}


def _filler_word(w: str) -> bool:
    return w in FILLERS or w in SOFT or bool(_HESITATION.match(w))


def is_noise(text: str, *, expecting_reply: bool = False) -> bool:
    """True if `text` is only a hesitation, a reaction or an invented filler word, so the tutor should stay quiet instead of answering it.
    With `expecting_reply` (a question or a 'did you mean' is waiting) short replies such as "yes" or "okay" are real and are kept."""
    w = _words(text)
    if not w:
        return True
    j = " ".join(w)
    if expecting_reply and j in {"yes", "yeah", "ok", "okay", "i", "so", "sure"}:
        return False
    if j in FILLERS:
        return True
    for phrase in sorted((f for f in FILLERS if " " in f), key=len, reverse=True):      # "let me think", "one second": multi-word fillers
        j = re.sub(rf"\b{re.escape(phrase)}\b", " ", j)
    rest = j.split()
    return len(w) <= 8 and all(_filler_word(x) for x in rest)
