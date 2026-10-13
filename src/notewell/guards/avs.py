"""After-visit summary checks: readable for the patient, and no number or drug that the note does not contain."""

from __future__ import annotations

import re

from notewell.guards.grounding import drugs_in
from notewell.guards.text import numbers
from notewell.models import Flag

TARGET_GRADE = 8.0


def _syllables(word: str) -> int:
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1 and not word.endswith("le"):
        count -= 1
    return max(1, count)


def reading_grade(text: str) -> float:
    """Flesch-Kincaid grade level."""
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return 0.0
    syllables = sum(_syllables(w) for w in words)
    return round(0.39 * len(words) / sentences + 11.8 * syllables / len(words) - 15.59, 1)


def check_avs(avs: str, *, note_text: str, transcript_text: str) -> list[Flag]:
    flags: list[Flag] = []
    if not avs.strip():
        return [Flag(kind="readability", severity="warning", message="No after-visit summary was drafted.")]
    grade = reading_grade(avs)
    if grade > TARGET_GRADE + 2:
        flags.append(
            Flag(
                kind="readability",
                severity="warning",
                message=f"Reading level is grade {grade:g}; aim for grade {TARGET_GRADE:g} or lower.",
                detail={"grade": grade},
            )
        )
    allowed_numbers = numbers(note_text) | numbers(transcript_text)
    extra_numbers = sorted(n for n in numbers(avs) - allowed_numbers if n not in {"1", "2", "911", "988"})
    if extra_numbers:
        flags.append(
            Flag(
                kind="unsupported",
                severity="error",
                message=f"The summary contains numbers that are not in the note or the transcript: {', '.join(extra_numbers)}.",
                detail={"numbers": extra_numbers},
            )
        )
    extra_drugs = sorted(drugs_in(avs) - drugs_in(note_text) - drugs_in(transcript_text))
    if extra_drugs:
        flags.append(
            Flag(
                kind="unsupported",
                severity="error",
                message=f"The summary mentions medications that are not in the note: {', '.join(extra_drugs)}.",
                detail={"drugs": extra_drugs},
            )
        )
    return flags
