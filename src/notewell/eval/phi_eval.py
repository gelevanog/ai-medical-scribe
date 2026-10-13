"""PHI detection against the gold identifiers of the synthetic visits, and the canary leak test.

Coverage of a gold value: every alphanumeric character of each occurrence (titles such as "Dr." excluded) lies inside
a masked span. A value counts as leaked if any occurrence is not fully covered.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from notewell.dataset import Case
from notewell.models import Turn
from notewell.phi.detector import Span
from notewell.phi.masker import Masker, registered_from

_TITLE = re.compile(r"^(?:Dr|Doctor|Mr|Mrs|Ms|Miss|Mx|Nurse|Coach|Professor)\.?\s+")


@dataclass
class PHIResult:
    total: int = 0
    covered: int = 0
    by_type: Counter[str] = field(default_factory=Counter)
    covered_by_type: Counter[str] = field(default_factory=Counter)
    misses: list[dict[str, str]] = field(default_factory=list)
    masked_tokens: int = 0
    masked_tokens_not_phi: int = 0
    false_positive_examples: Counter[str] = field(default_factory=Counter)

    @property
    def recall(self) -> float:
        return self.covered / self.total if self.total else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "gold_values": self.total,
            "covered": self.covered,
            "recall": round(self.recall, 4),
            "by_type": {k: {"total": v, "covered": self.covered_by_type[k]} for k, v in sorted(self.by_type.items())},
            "misses": self.misses,
            "masked_words": self.masked_tokens,
            "masked_words_not_in_gold_phi": self.masked_tokens_not_phi,
            "over_masking_rate": round(self.masked_tokens_not_phi / self.masked_tokens, 4) if self.masked_tokens else 0,
            "over_masked_examples": dict(self.false_positive_examples.most_common(40)),
        }


def _covered(text: str, start: int, end: int, spans: list[Span]) -> bool:
    for i in range(start, end):
        if text[i].isalnum() and not any(s.start <= i < s.end for s in spans):
            return False
    return True


def evaluate_case(case: Case, turns: list[Turn], masker: Masker, result: PHIResult) -> None:
    registered = registered_from(case.patient, case.clinician.name)
    masked = masker.mask_turns(turns, registered)
    gold_char_sets: list[set[int]] = [set() for _ in turns]
    for phi in case.gold.phi:
        value = _TITLE.sub("", phi.value).strip()
        if not value:
            continue
        found_any = False
        ok = True
        for ti, turn in enumerate(turns):
            for m in re.finditer(re.escape(value), turn.text, re.IGNORECASE):
                found_any = True
                gold_char_sets[ti].update(range(m.start(), m.end()))
                if not _covered(turn.text, m.start(), m.end(), masked.spans[ti]):
                    ok = False
        if not found_any:
            continue  # ASR changed the surface form: not countable here (the leak test covers it)
        result.total += 1
        result.by_type[phi.type] += 1
        if ok:
            result.covered += 1
            result.covered_by_type[phi.type] += 1
        else:
            result.misses.append({"case": case.id, "type": phi.type, "value": phi.value})
    for ti, turn in enumerate(turns):
        for span in masked.spans[ti]:
            for w in re.finditer(r"[A-Za-z0-9]+", turn.text[span.start : span.end]):
                result.masked_tokens += 1
                pos = span.start + w.start()
                if pos not in gold_char_sets[ti]:
                    result.masked_tokens_not_phi += 1
                    result.false_positive_examples[f"{span.type}:{w.group(0)}"] += 1


def evaluate(cases: list[Case], masker: Masker | None = None, *, turns_by_case: dict[str, list[Turn]] | None = None) -> PHIResult:
    masker = masker or Masker()
    result = PHIResult()
    for case in cases:
        turns = turns_by_case[case.id] if turns_by_case else case.dialogue
        evaluate_case(case, turns, masker, result)
    return result
