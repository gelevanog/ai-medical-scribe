"""Scoring a drafted note against the gold annotations of a synthetic visit.

* Fact recall: a gold fact is found when one note statement contains every keyword group of the fact (any synonym)
  with the same polarity ("Denies chest pain" for an absent fact). Section-correct recall also needs the section.
* Negation errors: a statement matches a gold fact's keywords with the opposite polarity, and no statement states it
  correctly. Trap errors: a statement asserts a trap (family history, hypothetical, another person, a corrected value)
  as the patient's own current fact.
* Gold-matched precision: the share of statements that match at least one gold fact (a lower bound: correct details
  that the gold list does not contain count against it). Unsupported rate: statements the grounding guard could not
  tie to the transcript.
* Codes: primary ICD-10-CM in the top 1 / top 3 (exact; lenient = a listed alternative; category = first 3 characters),
  recall of all gold diagnoses, E/M exact and within one level, CPT procedure recall.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from notewell.codes.em import code_level
from notewell.dataset import Case, GoldFact
from notewell.guards.medications import normalize_drug
from notewell.guards.negation import clause_polarity
from notewell.guards.text import FAMILY, HYPOTHETICAL, OTHER_PEOPLE, numbers
from notewell.models import Note

_LEVELS = ["straightforward", "low", "moderate", "high"]


def _norm(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"(\d+)\s*/\s*(\d+)", r"\1 over \2", t)
    return re.sub(r"\s+", " ", t)


def _group_hit(text: str, group: list[str]) -> re.Match[str] | None:
    """Any synonym of the group, on a left word boundary (so stems like "wheez" match "wheezing")."""
    for syn in group:
        m = re.search(r"(?<![a-z0-9])" + re.escape(_norm(syn)), text)
        if m:
            return m
    return None


def match_fact(statement: str, groups: list[list[str]]) -> re.Match[str] | None:
    text = _norm(statement)
    first: re.Match[str] | None = None
    for group in groups:
        m = _group_hit(text, group)
        if m is None:
            return None
        first = first or m
    return first


_POST_NEGATION = re.compile(r"^\W*(?:\w+\W+){0,3}?(?:not|never|negative|absent|none)\b")


def statement_polarity(statement: str, match: re.Match[str]) -> str:
    """present | absent | hypothetical for the matched finding: NegEx-style triggers before it in the same clause,
    a short post-negation ("thyroid not enlarged", "never had one"), or a precaution ("return if ...")."""
    text = _norm(statement)
    pol = clause_polarity(text, match.start(), match.end(), others=False)
    if pol == "hypothetical":
        return "hypothetical"
    if pol == "absent" or _POST_NEGATION.match(text[match.end() :]):
        return "absent"
    return "present"


def note_lines(note: Note) -> list[tuple[str, str]]:
    """(section, text) of every statement, plus medications and allergies as subjective lines."""
    lines: list[tuple[str, str]] = [(s.section, s.text) for s in note.statements]
    lines += [("subjective", f"{m.name} {m.dose} {m.frequency} {m.status}") for m in note.medications]
    lines += [("subjective", f"allergy to {a.substance} {a.reaction}") for a in note.allergies]
    return lines


@dataclass
class CaseScore:
    case: str
    specialty: str
    facts: int = 0
    found: int = 0
    found_in_section: int = 0
    negation_errors: list[str] = field(default_factory=list)
    trap_errors: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    statements: int = 0
    statements_matching_gold: int = 0
    unsupported: int = 0
    partial: int = 0
    flags: Counter[str] = field(default_factory=Counter)
    meds: int = 0
    meds_found: int = 0
    meds_dose_correct: int = 0
    icd_primary: str = ""
    icd_suggested: list[str] = field(default_factory=list)
    icd_top1: bool = False
    icd_top3: bool = False
    icd_top3_lenient: bool = False
    icd_top3_category: bool = False
    icd_gold_recall: float = 0.0
    icd_invalid: int = 0
    em_gold: str = ""
    em_suggested: str = ""
    em_exact: bool = False
    em_within_one: bool = False
    em_rule_estimate: str = ""
    procedures_gold: int = 0
    procedures_found: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["flags"] = dict(self.flags)
        return d


def score_note(case: Case, note: Note) -> CaseScore:
    score = CaseScore(case=case.id, specialty=case.specialty)
    lines = note_lines(note)
    score.statements = len(note.statements)
    matched_statements: set[int] = set()
    for fact in case.gold.facts:
        score.facts += 1
        _score_fact(fact, lines, score, matched_statements)
    score.statements_matching_gold = len({i for i in matched_statements if i < len(note.statements)})
    legit = {lines[i][1] for i in matched_statements}
    for trap in case.traps:
        for _, text in lines:
            if text in legit:
                continue  # a statement that states a gold fact is not a trap error ("salt at a wedding" is the cause)
            m = match_fact(text, trap.match)
            if m is None:
                continue
            lowered = text.lower()
            hedged = bool(HYPOTHETICAL.search(lowered) or FAMILY.search(lowered) or OTHER_PEOPLE.search(lowered))
            if trap.type == "correction":
                hedged = bool(re.search(r"\b(?:corrected|not|previously|was|earlier|from)\b", lowered))
            if trap.type == "past_resolved":
                hedged = hedged or bool(
                    re.search(r"\b(?:resolved|history of|previous|prior|past|ago|was|had)\b", lowered)
                )
            if trap.type == "small_talk":
                hedged = False
            if not hedged and statement_polarity(text, m) == "present":
                score.trap_errors.append(f"{trap.type}: {trap.text}")
                break
    for s in note.statements:
        if s.support == "unsupported":
            score.unsupported += 1
        elif s.support == "partial":
            score.partial += 1
        for f in s.flags:
            score.flags[f.kind] += 1
    for med in note.medications:
        for f in med.flags:
            score.flags[f.kind] += 1
    # medications
    for gm in case.gold.medications:
        score.meds += 1
        gold_drug = normalize_drug(gm.name)
        for med in note.medications:
            drug = normalize_drug(med.name)
            same = (
                drug is not None and gold_drug is not None and drug.name == gold_drug.name
            ) or med.name.lower() == gm.name.lower()
            if same:
                score.meds_found += 1
                if not gm.dose or (numbers(gm.dose) & numbers(f"{med.dose}")):
                    score.meds_dose_correct += 1
                break
    # ICD-10-CM
    primary = case.gold.primary
    suggested = [c.code for c in note.icd10]
    score.icd_primary = primary.code
    score.icd_suggested = suggested
    score.icd_invalid = sum(1 for c in note.icd10 if not c.valid)
    top3 = suggested[:3]
    lenient = {primary.code, *primary.alternatives}
    score.icd_top1 = bool(suggested) and suggested[0] == primary.code
    score.icd_top3 = primary.code in top3
    score.icd_top3_lenient = bool(lenient & set(top3))
    score.icd_top3_category = any(c[:3] == primary.code[:3] for c in top3)
    gold_codes = list(case.gold.diagnoses)
    hits = sum(1 for d in gold_codes if ({d.code, *d.alternatives} & set(suggested)))
    score.icd_gold_recall = hits / len(gold_codes) if gold_codes else 1.0
    # E/M
    score.em_gold = case.gold.em_code
    score.em_suggested = note.em.code if note.em else ""
    score.em_rule_estimate = note.mdm.code if note.mdm else ""
    score.em_exact = score.em_suggested == score.em_gold
    g, s_ = code_level(score.em_gold), code_level(score.em_suggested)
    same_family = score.em_gold[:4] == score.em_suggested[:4]
    score.em_within_one = score.em_exact or bool(
        g and s_ and same_family and abs(_LEVELS.index(g) - _LEVELS.index(s_)) <= 1
    )
    gold_proc = {str(p["code"]) for p in case.gold.procedures}
    score.procedures_gold = len(gold_proc)
    score.procedures_found = len(gold_proc & {p.code for p in note.procedures})
    return score


def _score_fact(fact: GoldFact, lines: list[tuple[str, str]], score: CaseScore, matched: set[int]) -> None:
    correct = False
    in_section = False
    wrong_polarity = False
    for i, (section, text) in enumerate(lines):
        m = match_fact(text, fact.match)
        if m is None:
            continue
        matched.add(i)
        polarity = statement_polarity(text, m)
        if polarity == fact.polarity or (polarity == "hypothetical" and fact.section == "plan"):
            correct = True
            in_section = in_section or section == fact.section
        elif polarity != "hypothetical" and section == fact.section and fact.section in {"subjective", "objective"}:
            # A finding stated with the opposite polarity in the same section ("Denies dizziness" for a reported
            # dizziness). Plan instructions ("no basketball for 2 weeks") are not findings.
            wrong_polarity = True
    if correct:
        score.found += 1
        score.found_in_section += in_section
    else:
        score.missed.append(f"{fact.id} [{fact.section}] {fact.text}")
        if wrong_polarity:
            score.negation_errors.append(f"{fact.id}: {fact.text} ({fact.polarity})")


def aggregate(scores: list[CaseScore]) -> dict[str, Any]:
    def ratio(a: float, b: float) -> float | None:
        return round(a / b, 4) if b else None

    n = len(scores)
    facts = sum(s.facts for s in scores)
    statements = sum(s.statements for s in scores)
    flags: Counter[str] = Counter()
    for s in scores:
        flags.update(s.flags)
    by_specialty: dict[str, dict[str, Any]] = {}
    for spec in sorted({s.specialty for s in scores}):
        group = [s for s in scores if s.specialty == spec]
        by_specialty[spec] = {
            "visits": len(group),
            "fact_recall": ratio(sum(s.found for s in group), sum(s.facts for s in group)),
            "icd_top3": ratio(sum(s.icd_top3 for s in group), len(group)),
            "unsupported_rate": ratio(sum(s.unsupported for s in group), sum(s.statements for s in group)),
        }
    return {
        "visits": n,
        "gold_facts": facts,
        "fact_recall": ratio(sum(s.found for s in scores), facts),
        "fact_recall_section_correct": ratio(sum(s.found_in_section for s in scores), facts),
        "statements": statements,
        "gold_matched_precision": ratio(sum(s.statements_matching_gold for s in scores), statements),
        "unsupported_rate": ratio(sum(s.unsupported for s in scores), statements),
        "partial_rate": ratio(sum(s.partial for s in scores), statements),
        "negation_errors": sum(len(s.negation_errors) for s in scores),
        "negatives_in_gold": None,
        "trap_errors": sum(len(s.trap_errors) for s in scores),
        "medication_recall": ratio(sum(s.meds_found for s in scores), sum(s.meds for s in scores)),
        "medication_dose_correct": ratio(sum(s.meds_dose_correct for s in scores), sum(s.meds for s in scores)),
        "icd_top1": ratio(sum(s.icd_top1 for s in scores), n),
        "icd_top3": ratio(sum(s.icd_top3 for s in scores), n),
        "icd_top3_lenient": ratio(sum(s.icd_top3_lenient for s in scores), n),
        "icd_top3_category": ratio(sum(s.icd_top3_category for s in scores), n),
        "icd_all_gold_recall": ratio(sum(s.icd_gold_recall for s in scores), n),
        "icd_invalid_suggestions": sum(s.icd_invalid for s in scores),
        "em_exact": ratio(sum(s.em_exact for s in scores), n),
        "em_within_one": ratio(sum(s.em_within_one for s in scores), n),
        "em_rule_estimate_exact": ratio(sum(s.em_rule_estimate == s.em_gold for s in scores), n),
        "procedure_recall": ratio(sum(s.procedures_found for s in scores), sum(s.procedures_gold for s in scores)),
        "flags": dict(flags),
        "by_specialty": by_specialty,
    }
