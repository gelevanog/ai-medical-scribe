"""Text normalization shared by the guards: tokens, number words, light stemming, negation cues."""

from __future__ import annotations

import re
from functools import lru_cache

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "half": 0.5, "a couple": 2,
    "once": 1, "twice": 2,
}  # fmt: skip
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_UNITS = {k: v for k, v in _NUMBER_WORDS.items() if isinstance(v, int) and 1 <= v <= 9}

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "my",
        "your",
        "his",
        "her",
        "our",
        "their",
        "me",
        "him",
        "us",
        "them",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "having",
        "not",
        "no",
        "so",
        "if",
        "then",
        "than",
        "there",
        "here",
        "about",
        "into",
        "over",
        "under",
        "up",
        "down",
        "out",
        "off",
        "just",
        "also",
        "very",
        "really",
        "some",
        "any",
        "all",
        "can",
        "could",
        "would",
        "should",
        "will",
        "may",
        "might",
        "must",
        "shall",
        "per",
        "day",
        "days",
        "week",
        "weeks",
        "month",
        "months",
        "year",
        "years",
        "daily",
        "patient",
        "pt",
        "reports",
        "report",
        "reported",
        "states",
        "stated",
        "says",
        "said",
        "notes",
        "noted",
        "denies",
        "denied",
        "endorses",
        "endorsed",
        "describes",
        "described",
        "feels",
        "feel",
        "felt",
        "has",
        "having",
        "currently",
        "today",
        "now",
        "recently",
        "since",
        "ago",
        "last",
        "past",
        "currently",
        "mild",
        "moderate",
        "severe",
    ]
)

NEGATION_PRE = re.compile(
    r"\b(?:no|not|denies|denied|deny|denying|without|negative for|free of|absence of|never|none|nor|"
    r"doesn't|does not|didn't|did not|don't|do not|isn't|is not|wasn't|was not|haven't|have not|hasn't|has not|"
    r"hadn't|had not|can't|cannot|won't|nothing|no longer|rules out|ruled out|unremarkable for|no signs? of|"
    r"no evidence of|no history of|no h/o|w/o)\b",
    re.IGNORECASE,
)
NEGATION_POST = re.compile(r"\b(?:is|are|was|were)?\s*(?:negative|absent|ruled out|resolved|gone|not present)\b", re.I)
SCOPE_BREAK = re.compile(
    r"\b(?:but|however|although|though|except|aside from|apart from|other than|yet|which|who)\b|[.;:!?]", re.I
)
FAMILY = re.compile(
    r"\b(?:mother|mom|father|dad|brother|sister|sibling|grandmother|grandma|grandfather|grandpa|aunt|uncle|cousin|"
    r"parents?|family history|family hx|runs in (?:the|my|our|his|her) family)\b",
    re.I,
)
OTHER_PEOPLE = re.compile(
    r"\b(?:wife|husband|spouse|partner|boyfriend|girlfriend|son|daughter|kids?|children|friend|coworker|co-worker|"
    r"neighbor|roommate|boss|classmates?|half the room|everyone at)\b",
    re.I,
)
HYPOTHETICAL = re.compile(
    r"\b(?:if|in case|should (?:you|it|she|he|they)|unless|watch for|look out for|call (?:us )?if|come back if|"
    r"return if|go to the (?:er|emergency)|precautions?|would|could|might|may need|consider|considering|"
    r"plan to|planning|thinking about|down the road|in the future|eventually|possible|possibly)\b",
    re.I,
)
AFFIRM = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|uh-huh|mm-hmm|mhm|sure|definitely|absolutely|a little|a bit|sometimes|some|kind of|"
    r"sort of|i do|i have|it does|it is|correct|right|actually yes|occasionally|once or twice)\b",
    re.I,
)
DENY = re.compile(
    r"^\s*(?:no|nope|nah|not really|not at all|never|none|nothing|i don't|i do not|i haven't|i have not|"
    r"she doesn't|he doesn't|she hasn't|he hasn't|she didn't|he didn't|it doesn't|it hasn't|not that i know|"
    r"not that i've noticed|not since|no,|nope,|uh-uh|negative)\b",
    re.I,
)


def number_value(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def words_to_digits(text: str) -> str:
    """ "two weeks" -> "2 weeks", "twenty-five" -> "25", "a hundred and two" -> "102"."""

    def tens(match: re.Match[str]) -> str:
        return str(_TENS[match.group(1).lower()] + _UNITS[match.group(2).lower()])

    out = re.sub(r"\b(" + "|".join(_TENS) + r")[- ](" + "|".join(_UNITS) + r")\b", tens, text, flags=re.IGNORECASE)
    out = re.sub(r"\b(?:a|one) hundred(?: and)? (\d+)\b", lambda m: str(100 + int(m.group(1))), out, flags=re.I)

    def single(match: re.Match[str]) -> str:
        value = _NUMBER_WORDS[match.group(0).lower()]
        return str(value)

    keys = sorted((k for k in _NUMBER_WORDS if " " not in k), key=len, reverse=True)
    return re.sub(r"\b(?:" + "|".join(keys) + r")\b", single, out, flags=re.IGNORECASE)


_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def stem(word: str) -> str:
    for suffix in ("ing", "edly", "ed", "es", "s", "ly", "al", "ic", "y"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


@lru_cache(maxsize=8192)
def tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(words_to_digits(text.lower()).replace("/", " ")))


def content_terms(text: str) -> set[str]:
    return {stem(t) for t in tokens(text) if t not in STOPWORDS and (len(t) >= 3 or t.replace(".", "").isdigit())}


def numbers(text: str) -> set[str]:
    """Numbers that carry meaning (doses, vitals, durations), normalized: "1,440" -> "1440", "10.0" -> "10"."""
    out: set[str] = set()
    for t in tokens(text):
        value = number_value(t)
        if value is None:
            continue
        out.add(str(int(value)) if value == int(value) else str(value))
    return out


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\"”']?\s+(?=[\"“']?[A-Z0-9])")


def sentences(text: str) -> list[tuple[int, str]]:
    """(offset, sentence) pairs; decimals ("0.9") and abbreviations followed by lowercase do not split."""
    out: list[tuple[int, str]] = []
    start = 0
    for m in _SENTENCE_SPLIT.finditer(text):
        piece = text[start : m.start()]
        if re.search(r"\b(?:Dr|Mr|Mrs|Ms|St|vs|No)\.$", piece):
            continue
        if piece.strip():
            out.append((start, piece))
        start = m.end()
    if text[start:].strip():
        out.append((start, text[start:]))
    return out
