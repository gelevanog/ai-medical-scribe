"""Medication-name correction after speech recognition: Whisper writes rare drug names the way they sound
("adorvastatin", "Rasuvastatin"). A word that is not an English word and is within a small edit distance of exactly
one drug name in the drug list (same first letter) is replaced by it."""

from __future__ import annotations

import re
from functools import lru_cache

from notewell.kb import drug_aliases
from notewell.phi.detector import _levenshtein, lexicons

_WORD = re.compile(r"[A-Za-z][A-Za-z-]{5,}")


@lru_cache(maxsize=1)
def _names() -> tuple[str, ...]:
    return tuple(sorted({w for alias in drug_aliases() for w in alias.split() if len(w) >= 6 and w.isalpha()}))


def correct_drug_names(text: str) -> tuple[str, int]:
    english = lexicons()[3]
    names = _names()
    count = 0

    def fix(match: re.Match[str]) -> str:
        nonlocal count
        word = match.group(0)
        lw = word.lower()
        if lw in english or lw in names:
            return word
        limit = 1 if len(lw) < 8 else 2
        candidates = [
            n for n in names if n[0] == lw[0] and abs(len(n) - len(lw)) <= limit and _levenshtein(lw, n, limit) <= limit
        ]
        if len(candidates) != 1:
            return word
        count += 1
        fixed = candidates[0]
        return fixed.capitalize() if word[0].isupper() else fixed

    return _WORD.sub(fix, text), count
