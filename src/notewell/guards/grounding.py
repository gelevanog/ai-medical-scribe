"""Claim-to-quote grounding: every note statement must be backed by what was said.

For each statement:

1. **Quote check**: each evidence quote the model gave must occur (fuzzily, token by token) in the turn it cites, or
   in another turn, which then replaces the citation. Unverifiable quotes are dropped.
2. **Content check**: the numbers and drug names in the statement must occur in the evidence turns (or the adjacent
   question/answer turn); a dose or a vital sign that nobody said is the most dangerous kind of hallucination.
3. **Auto-link**: a statement without usable evidence is linked to the best-matching turn if its content terms,
   numbers and drugs are all found there.

Result: supported (verified quote and content), partial (quote verified but a number/drug or most content terms not
found), unsupported (nothing in the transcript backs it). Partial and unsupported statements carry flags.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from notewell.guards.text import content_terms, numbers, tokens
from notewell.kb import drug_aliases, drug_pattern
from notewell.models import Evidence, Flag, Statement, Turn

QUOTE_MIN_RATIO = 0.8
CONTENT_MIN_OVERLAP = 0.34
AUTOLINK_MIN_OVERLAP = 0.6


@dataclass
class TurnText:
    idx: int
    speaker: str
    text: str
    tokens: tuple[str, ...]
    terms: set[str]
    numbers: set[str]
    drugs: set[str]


def drugs_in(text: str) -> set[str]:
    aliases = drug_aliases()
    return {aliases[m.group(1).lower()] for m in drug_pattern().finditer(text)}


def _fuzzy_contains(quote: tuple[str, ...], turn: tuple[str, ...]) -> float:
    """Fraction of the quote's tokens found in order in the turn (longest common subsequence): tolerates the small
    word changes of speech recognition and of models that tidy up a quote."""
    if not quote or not turn:
        return 0.0
    prev = [0] * (len(turn) + 1)
    for q in quote:
        cur = [0]
        for j, t in enumerate(turn, 1):
            cur.append(prev[j - 1] + 1 if q == t else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1] / len(quote)


class Grounder:
    def __init__(self, turns: Sequence[Turn]) -> None:
        self.turns = [
            TurnText(t.idx, t.speaker, t.text, tokens(t.text), content_terms(t.text), numbers(t.text), drugs_in(t.text))
            for t in turns
        ]
        self.by_idx = {t.idx: t for t in self.turns}

    def neighbors(self, idx: int) -> list[TurnText]:
        """The cited turn plus the turn before and after (a question and its answer belong together)."""
        return [self.by_idx[i] for i in (idx - 1, idx, idx + 1) if i in self.by_idx]

    def verify_quote(self, ev: Evidence) -> Evidence | None:
        quote = tokens(ev.quote)
        if not quote:
            if ev.turn in self.by_idx:
                return Evidence(turn=ev.turn, quote="", verified=False, score=0.0)
            return None
        cited = self.by_idx.get(ev.turn)
        if cited is not None:
            score = _fuzzy_contains(quote, cited.tokens)
            if score >= QUOTE_MIN_RATIO:
                return Evidence(turn=cited.idx, quote=ev.quote, verified=True, score=round(score, 2))
        best: tuple[float, TurnText | None] = (0.0, None)
        for t in self.turns:
            score = _fuzzy_contains(quote, t.tokens)
            if score > best[0]:
                best = (score, t)
        if best[1] is not None and best[0] >= QUOTE_MIN_RATIO:
            return Evidence(turn=best[1].idx, quote=ev.quote, verified=True, score=round(best[0], 2))
        return None

    def ground(self, statement: Statement) -> Statement:
        evidence = [e for e in (self.verify_quote(ev) for ev in statement.evidence) if e is not None]
        verified = [e for e in evidence if e.verified]
        text = statement.text
        stmt_numbers = numbers(text)
        stmt_drugs = drugs_in(text)
        stmt_terms = content_terms(text) - stmt_numbers
        flags: list[Flag] = []

        def coverage(turn_ids: set[int]) -> tuple[set[str], set[str], float]:
            pool: list[TurnText] = []
            for idx in turn_ids:
                pool.extend(self.neighbors(idx))
            nums = set().union(*(t.numbers for t in pool)) if pool else set()
            drugs = set().union(*(t.drugs for t in pool)) if pool else set()
            terms = set().union(*(t.terms for t in pool)) if pool else set()
            missing_nums = stmt_numbers - nums
            missing_drugs = stmt_drugs - drugs
            overlap = len(stmt_terms & terms) / len(stmt_terms) if stmt_terms else 1.0
            return missing_nums, missing_drugs, overlap

        support = "unsupported"
        if verified:
            missing_nums, missing_drugs, overlap = coverage({e.turn for e in verified})
            if not missing_nums and not missing_drugs and overlap >= CONTENT_MIN_OVERLAP:
                support = "supported"
            else:
                # The model may cite the wrong turn: look for a better one before flagging.
                linked = self._autolink(stmt_terms, stmt_numbers, stmt_drugs)
                if linked is not None:
                    evidence.append(Evidence(turn=linked.idx, quote=_snippet(linked.text), verified=True, score=0.0))
                    support = "supported"
                else:
                    support = "partial"
                    detail: dict[str, object] = {}
                    if missing_nums:
                        detail["numbers_not_in_evidence"] = sorted(missing_nums)
                    if missing_drugs:
                        detail["drugs_not_in_evidence"] = sorted(missing_drugs)
                    severity = "error" if missing_nums or missing_drugs else "warning"
                    what = ", ".join(sorted(missing_nums | missing_drugs)) or "most of its terms"
                    flags.append(
                        Flag(
                            kind="partial_support",
                            severity=severity,
                            message=f"Quoted turn does not contain {what}; check against the transcript.",
                            detail=detail,
                        )
                    )
        else:
            linked = self._autolink(stmt_terms, stmt_numbers, stmt_drugs)
            if linked is not None:
                evidence = [Evidence(turn=linked.idx, quote=_snippet(linked.text), verified=True, score=0.0)]
                support = "supported"
            else:
                flags.append(
                    Flag(
                        kind="unsupported",
                        severity="error",
                        message="No supporting quote found in the transcript. Remove it or confirm it yourself.",
                    )
                )
        return statement.model_copy(update={"evidence": evidence, "support": support, "flags": [*statement.flags, *flags]})

    def _autolink(self, terms: set[str], nums: set[str], drugs: set[str]) -> TurnText | None:
        if not terms and not nums:
            return None
        best: tuple[float, TurnText | None] = (0.0, None)
        for t in self.turns:
            pool = self.neighbors(t.idx)
            pool_terms = set().union(*(p.terms for p in pool))
            pool_nums = set().union(*(p.numbers for p in pool))
            pool_drugs = set().union(*(p.drugs for p in pool))
            if nums - pool_nums or drugs - pool_drugs:
                continue
            overlap = len(terms & t.terms) / len(terms) if terms else 1.0
            overlap_pool = len(terms & pool_terms) / len(terms) if terms else 1.0
            score = overlap + 0.25 * overlap_pool
            if overlap_pool >= AUTOLINK_MIN_OVERLAP and score > best[0]:
                best = (score, t)
        return best[1]


def _snippet(text: str, words: int = 14) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + (" ..." if len(parts) > words else "")
