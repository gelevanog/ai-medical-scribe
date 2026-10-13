"""Speech recognition and diarization metrics against the gold dialogue text.

* WER with a normalizer that treats "two weeks" and "2 weeks", "152/94" and "152 over 94", "mm-hmm" spellings alike.
* Key-term recall: drug names and numbers of the gold text that the recognizer got exactly right (aligned).
* Word-level role accuracy: for every aligned word, is the hypothesis speaker role the gold role.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from notewell.guards.text import words_to_digits
from notewell.kb import drug_aliases
from notewell.models import Turn

_FILLERS = {"um", "uh", "hmm", "mm", "mhm", "er", "ah"}


def normalize(text: str) -> list[str]:
    t = text.lower().replace("’", "'")
    t = re.sub(r"\bmm-?hmm\b|\bmhm\b|\buh-huh\b", "mhm", t)
    t = re.sub(r"(\d+)\s*/\s*(\d+)", r"\1 over \2", t)
    t = t.replace("%", " percent").replace("&", " and ")
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    t = words_to_digits(t)
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", t)
    t = re.sub(r"[^a-z0-9.' ]+", " ", t.replace("-", " "))
    words = [w.strip(".'") for w in t.split()]
    return [w for w in words if w and w not in _FILLERS]


@dataclass
class Alignment:
    pairs: list[tuple[int | None, int | None]]
    substitutions: int
    deletions: int
    insertions: int


def align(ref: list[str], hyp: list[str]) -> Alignment:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
    i, j = n, m
    pairs: list[tuple[int | None, int | None]] = []
    s = d = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                s += 1
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            pairs.append((i - 1, None))
            i -= 1
        else:
            ins += 1
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return Alignment(pairs, s, d, ins)


@dataclass
class ASRScore:
    ref_words: int = 0
    errors: int = 0
    key_terms: int = 0
    key_terms_correct: int = 0
    role_words: int = 0
    role_correct: int = 0
    misrecognized_terms: dict[str, int] = field(default_factory=dict)

    @property
    def wer(self) -> float:
        return self.errors / self.ref_words if self.ref_words else 0.0

    def add(self, other: ASRScore) -> None:
        self.ref_words += other.ref_words
        self.errors += other.errors
        self.key_terms += other.key_terms
        self.key_terms_correct += other.key_terms_correct
        self.role_words += other.role_words
        self.role_correct += other.role_correct
        for k, v in other.misrecognized_terms.items():
            self.misrecognized_terms[k] = self.misrecognized_terms.get(k, 0) + v

    def to_dict(self) -> dict[str, object]:
        return {
            "ref_words": self.ref_words,
            "wer": round(self.wer, 4),
            "key_terms": self.key_terms,
            "key_term_recall": round(self.key_terms_correct / self.key_terms, 4) if self.key_terms else None,
            "role_accuracy": round(self.role_correct / self.role_words, 4) if self.role_words else None,
        }


def _is_key(token: str) -> bool:
    return token.replace(".", "").isdigit() or token in _drug_tokens()


_DRUG_TOKENS: set[str] = set()


def _drug_tokens() -> set[str]:
    if not _DRUG_TOKENS:
        for alias in drug_aliases():
            _DRUG_TOKENS.update(w for w in re.findall(r"[a-z]+", alias) if len(w) >= 5)
    return _DRUG_TOKENS


def score(gold: list[Turn], hyp: list[Turn]) -> ASRScore:
    ref_tokens: list[str] = []
    ref_roles: list[str] = []
    for t in gold:
        toks = normalize(t.text)
        ref_tokens += toks
        ref_roles += [t.speaker] * len(toks)
    hyp_tokens: list[str] = []
    hyp_roles: list[str] = []
    for t in hyp:
        toks = normalize(t.text)
        hyp_tokens += toks
        hyp_roles += [t.speaker] * len(toks)
    a = align(ref_tokens, hyp_tokens)
    result = ASRScore(ref_words=len(ref_tokens), errors=a.substitutions + a.deletions + a.insertions)
    for ri, hi in a.pairs:
        if ri is None:
            continue
        token = ref_tokens[ri]
        if _is_key(token):
            result.key_terms += 1
            if hi is not None and hyp_tokens[hi] == token:
                result.key_terms_correct += 1
            elif not token.replace(".", "").isdigit():
                result.misrecognized_terms[token] = result.misrecognized_terms.get(token, 0) + 1
        if hi is not None:
            result.role_words += 1
            result.role_correct += ref_roles[ri] == hyp_roles[hi]
    return result
