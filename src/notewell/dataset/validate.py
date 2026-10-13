"""Consistency checks for a synthetic visit: every gold quote is verbatim in one turn, every identifier is spoken,
every code is valid in the official list, exactly one primary diagnosis, all four SOAP sections have facts."""

from __future__ import annotations

from notewell.codes.icd10 import CodeSet
from notewell.dataset import Case
from notewell.kb import cpt


def validate_case(case: Case, codeset: CodeSet) -> list[str]:
    errors: list[str] = []
    texts = [t.text for t in case.dialogue]

    def verbatim(quote: str, where: str) -> None:
        if quote and not any(quote in t for t in texts):
            errors.append(f"{where}: quote not verbatim in any turn: {quote[:60]!r}")

    for fact in case.gold.facts:
        verbatim(fact.evidence, f"fact {fact.id}")
    for med in case.gold.medications:
        verbatim(med.evidence, f"medication {med.name}")
    for dx in case.gold.diagnoses:
        verbatim(dx.evidence, f"diagnosis {dx.code}")
        for code in [dx.code, *dx.alternatives]:
            status = codeset.validate(code).status
            if status not in {"valid", "unverifiable"}:
                errors.append(f"diagnosis {code}: {status}")
    for trap in case.traps:
        verbatim(trap.evidence, f"trap {trap.type}")
    for phi in case.gold.phi:
        if not any(phi.value in t for t in texts):
            errors.append(f"identifier {phi.type} not spoken verbatim")
    if sum(1 for d in case.gold.diagnoses if d.primary) != 1:
        errors.append("exactly one primary diagnosis required")
    if {f.section for f in case.gold.facts} != {"subjective", "objective", "assessment", "plan"}:
        errors.append("facts must cover all four SOAP sections")
    catalog = cpt()
    if case.gold.em_code not in catalog:
        errors.append(f"E/M {case.gold.em_code} not in the CPT list")
    for p in case.gold.procedures:
        if str(p.get("code")) not in catalog:
            errors.append(f"procedure {p.get('code')} not in the CPT list")
    if len(set(case.speakers)) != 2:
        errors.append("exactly two speakers required")
    return errors
