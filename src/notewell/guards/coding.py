"""Coding guards: every suggested code is validated against the official list and tied to evidence.

ICD-10-CM: exists in the FY2027 code list, is billable (a category header gets its billable children listed), has a
verified quote, is consistent with the assessment, and does not code a finding the transcript denies or attributes to
a family member. E/M: the code family matches the visit type (new 99202-99205, established 99211-99215,
psychotherapy 9079x/9083x) and the level is compared with the deterministic MDM estimate. CPT procedures: only codes
from the allowed list, with evidence that the service happened today.
"""

from __future__ import annotations

from typing import Any

from notewell.codes.em import code_level
from notewell.codes.icd10 import CodeSet, dotted
from notewell.guards.grounding import Grounder
from notewell.guards.negation import ConceptIndex
from notewell.guards.text import content_terms
from notewell.kb import cpt, find_concepts
from notewell.models import CodeSuggestion, Flag, MDMEstimate, Statement
from notewell.scribe.parse import parse_evidence

_PSYCH = {"90791", "90792", "90832", "90834", "90837"}


def check_icd(
    raw: list[dict[str, Any]],
    *,
    codeset: CodeSet,
    grounder: Grounder,
    index: ConceptIndex,
    assessment: list[Statement],
    limit: int = 6,
) -> list[CodeSuggestion]:
    out: list[CodeSuggestion] = []
    seen: set[str] = set()
    assessment_terms = set().union(*(content_terms(s.text) for s in assessment)) if assessment else set()
    for rank, item in enumerate(raw[:limit], 1):
        code = dotted(str(item.get("code", "")))
        if not code or code in seen:
            continue
        seen.add(code)
        validation = codeset.validate(code)
        flags: list[Flag] = []
        if validation.status == "invalid":
            flags.append(Flag(kind="code_invalid", severity="error", message=validation.message))
        elif validation.status == "not_billable":
            flags.append(
                Flag(
                    kind="code_not_billable",
                    severity="error",
                    message=validation.message,
                    detail={"children": list(validation.children)},
                )
            )
        elif validation.status == "unverifiable":
            flags.append(Flag(kind="code_invalid", severity="warning", message=validation.message))
        evidence = [e for e in (grounder.verify_quote(ev) for ev in parse_evidence(item.get("evidence"))) if e]
        verified = [e for e in evidence if e.verified]
        title = validation.title or str(item.get("description", ""))
        if not verified:
            flags.append(
                Flag(kind="code_unsupported", severity="warning", message="No verified quote supports this code.")
            )
        if title and assessment_terms and not (content_terms(title) & assessment_terms):
            flags.append(
                Flag(
                    kind="code_unsupported",
                    severity="warning",
                    message="The code's title shares no terms with the assessment; check it matches a diagnosis.",
                )
            )
        for concept, _, _ in find_concepts(title):
            summary = index.summary(concept)
            if summary == "absent":
                flags.append(
                    Flag(
                        kind="code_negated",
                        severity="error",
                        message=f"The code describes {concept.replace('_', ' ')}, which the patient denied.",
                    )
                )
            elif summary == "family" and "family" not in title.lower():
                flags.append(
                    Flag(
                        kind="code_negated",
                        severity="error",
                        message=f"{concept.replace('_', ' ').capitalize()} was mentioned for a family member, "
                        "not the patient.",
                    )
                )
        out.append(
            CodeSuggestion(
                system="ICD-10-CM",
                code=validation.code if validation.status != "invalid" else code,
                description=title,
                rank=rank,
                rationale=str(item.get("rationale", "") or ""),
                evidence=evidence,
                valid=validation.status == "valid",
                billable=validation.status == "valid",
                flags=flags,
            )
        )
    return out


def check_em(raw: dict[str, Any] | None, *, visit_type: str, estimate: MDMEstimate) -> CodeSuggestion:
    code = str((raw or {}).get("code", "") or "").strip() or estimate.code
    rationale = str((raw or {}).get("rationale", "") or "")
    flags: list[Flag] = []
    catalog = cpt()
    entry = catalog.get(code)
    if entry is None or entry["group"] not in {"em_new", "em_established", "psychiatry"}:
        flags.append(Flag(kind="code_invalid", severity="error", message=f"{code} is not an office E/M code."))
    elif code not in _PSYCH and entry["group"] == "psychiatry":
        pass  # add-on psychotherapy codes ride along with an E/M code
    elif code not in _PSYCH:
        expected = "em_new" if visit_type == "new" else "em_established"
        if entry["group"] != expected:
            flags.append(
                Flag(
                    kind="em_visit_type",
                    severity="error",
                    message=f"{code} is for {'new' if entry['group'] == 'em_new' else 'established'} patients; this "
                    f"is a{'n established' if visit_type == 'established' else ' new'} patient visit.",
                )
            )
        level = code_level(code)
        if level and level != estimate.level:
            flags.append(
                Flag(
                    kind="em_mismatch",
                    severity="warning",
                    message=f"The rule-based MDM estimate is {estimate.level} ({estimate.code}): "
                    + "; ".join(estimate.reasons),
                    detail={"estimate": estimate.code},
                )
            )
    summary = entry["summary"] if entry else ""
    return CodeSuggestion(
        system="CPT",
        code=code,
        description=summary,
        rank=1,
        rationale=rationale,
        valid=entry is not None,
        billable=entry is not None,
        flags=flags,
    )


def check_procedures(raw: list[dict[str, Any]], *, grounder: Grounder) -> list[CodeSuggestion]:
    out: list[CodeSuggestion] = []
    catalog = cpt()
    for rank, item in enumerate(raw, 1):
        code = str(item.get("code", "")).strip()
        entry = catalog.get(code)
        flags: list[Flag] = []
        if entry is None or entry["group"] in {"em_new", "em_established"}:
            flags.append(Flag(kind="code_invalid", severity="error", message=f"{code} is not in the allowed CPT list."))
        evidence = [e for e in (grounder.verify_quote(ev) for ev in parse_evidence(item.get("evidence"))) if e]
        if not any(e.verified for e in evidence):
            flags.append(
                Flag(
                    kind="code_unsupported",
                    severity="warning",
                    message="No verified quote shows this service was performed today.",
                )
            )
        out.append(
            CodeSuggestion(
                system="CPT",
                code=code,
                description=entry["summary"] if entry else "",
                rank=rank,
                rationale=str(item.get("rationale", "") or ""),
                evidence=evidence,
                valid=entry is not None,
                billable=entry is not None,
                flags=flags,
            )
        )
    return out
