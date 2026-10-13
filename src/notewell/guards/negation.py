"""Polarity of clinical findings in the transcript and in note statements (a NegEx-style scope check plus the
question-and-answer rule that matters most in conversations).

"Any chest pain?" "No." negates chest pain although the patient never said the words. "My dad had a heart attack"
is family history. "If the rash spreads, come back" is a hypothetical. A note statement that asserts a finding the
transcript only ever denies, attributes to someone else or mentions as a hypothetical is flagged for the clinician.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from notewell.guards.text import (
    AFFIRM,
    DENY,
    FAMILY,
    HYPOTHETICAL,
    NEGATION_POST,
    NEGATION_PRE,
    OTHER_PEOPLE,
    SCOPE_BREAK,
)
from notewell.kb import find_concepts
from notewell.models import Turn

Polarity = Literal["present", "absent", "family", "hypothetical", "unknown"]

_SENTENCE = re.compile(r"[^.?!]+[.?!]?")


@dataclass
class Mention:
    concept: str
    turn: int
    polarity: Polarity
    surface: str


@dataclass
class ConceptIndex:
    mentions: dict[str, list[Mention]] = field(default_factory=lambda: defaultdict(list))

    def polarities(self, concept: str) -> set[Polarity]:
        return {m.polarity for m in self.mentions.get(concept, [])}

    def summary(self, concept: str) -> Polarity:
        pols = self.polarities(concept) - {"unknown"}
        if not pols:
            return "unknown"
        if "present" in pols:
            return "present"
        if pols == {"absent"}:
            return "absent"
        if "absent" in pols:
            return "absent"
        if pols == {"family"}:
            return "family"
        if pols == {"hypothetical"}:
            return "hypothetical"
        return "unknown"


def clause_polarity(sentence: str, start: int, end: int, *, others: bool = True) -> Polarity:
    """Polarity of the concept at [start, end) in one sentence: NegEx-style triggers within the same clause."""
    before = sentence[:start]
    breaks = list(SCOPE_BREAK.finditer(before))
    window = before[breaks[-1].end() :] if breaks else before
    window = window[-60:]
    after = sentence[end:]
    after_break = SCOPE_BREAK.search(after)
    after_window = after[: after_break.start()] if after_break else after
    if HYPOTHETICAL.search(window):
        return "hypothetical"
    if FAMILY.search(window) or FAMILY.search(after_window[:30]):
        return "family"
    if others and OTHER_PEOPLE.search(window):
        return "family"
    if NEGATION_PRE.search(window) or NEGATION_POST.match(after_window[:25]):
        return "absent"
    return "present"


def index_transcript(turns: list[Turn]) -> ConceptIndex:
    index = ConceptIndex()
    for i, turn in enumerate(turns):
        for sentence_match in _SENTENCE.finditer(turn.text):
            sentence = sentence_match.group(0)
            is_question = sentence.strip().endswith("?")
            for concept, start, end in find_concepts(sentence):
                # A caregiver's "my daughter" is the patient; a patient's "my daughter" is someone else.
                polarity = clause_polarity(sentence, start, end, others=turn.speaker == "patient")
                if is_question and turn.speaker == "clinician" and polarity in {"present", "absent"}:
                    polarity = _answer_polarity(turns, i)
                index.mentions[concept].append(Mention(concept, turn.idx, polarity, sentence[start:end]))
    return index


def _answer_polarity(turns: list[Turn], i: int) -> Polarity:
    """The next turn by the other speaker answers the clinician's question."""
    for nxt in turns[i + 1 : i + 3]:
        if nxt.speaker == turns[i].speaker:
            continue
        if DENY.match(nxt.text):
            return "absent"
        if AFFIRM.match(nxt.text):
            return "present"
        return "unknown"
    return "unknown"


def statement_polarity(text: str, start: int, end: int) -> Polarity:
    for sentence_match in _SENTENCE.finditer(text):
        if sentence_match.start() <= start < sentence_match.end():
            s0 = sentence_match.start()
            return clause_polarity(sentence_match.group(0), start - s0, end - s0, others=False)
    return clause_polarity(text, start, end, others=False)


@dataclass
class PolarityIssue:
    concept: str
    surface: str
    kind: Literal["negation", "attribution", "hypothetical"]
    statement_polarity: Polarity
    transcript_polarity: Polarity
    turns: list[int]


def check_statement(text: str, index: ConceptIndex, *, section: str = "") -> list[PolarityIssue]:
    issues: list[PolarityIssue] = []
    mentions_family = bool(FAMILY.search(text) or OTHER_PEOPLE.search(text))
    hedged = bool(HYPOTHETICAL.search(text)) or section == "plan"
    for concept, start, end in find_concepts(text):
        said = index.mentions.get(concept)
        if not said:
            continue
        mine = statement_polarity(text, start, end)
        theirs = index.summary(concept)
        turns = sorted({m.turn for m in said})
        if mine == "present" and theirs == "absent" and not mentions_family:
            issues.append(PolarityIssue(concept, text[start:end], "negation", mine, theirs, turns))
        elif mine == "absent" and theirs == "present":
            # "Denies fever" while the patient described a fever, unless the transcript also denies it somewhere.
            if "absent" not in index.polarities(concept):
                issues.append(PolarityIssue(concept, text[start:end], "negation", mine, theirs, turns))
        elif mine == "present" and theirs == "family" and not mentions_family:
            issues.append(PolarityIssue(concept, text[start:end], "attribution", mine, theirs, turns))
        elif mine == "present" and theirs == "hypothetical" and not hedged:
            issues.append(PolarityIssue(concept, text[start:end], "hypothetical", mine, theirs, turns))
    return issues
