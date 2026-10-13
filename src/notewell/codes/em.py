"""A deterministic estimate of the office E/M level (2021 AMA guidelines: medical decision making).

Two of three elements decide the level: the number and complexity of problems addressed, the amount and complexity
of data reviewed and analyzed, and the risk of patient management. The estimate reads the note's assessment and plan
(and the transcript for an independent historian). It is a cross-check for the model's suggestion, not a coder.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from notewell.models import MDMEstimate, Medication, Statement

LEVELS = ("straightforward", "low", "moderate", "high")
NEW = {"straightforward": "99202", "low": "99203", "moderate": "99204", "high": "99205"}
ESTABLISHED = {"straightforward": "99212", "low": "99213", "moderate": "99214", "high": "99215"}

_EXACERBATION = re.compile(
    r"\b(?:not at goal|not controlled|uncontrolled|poorly controlled|worse|worsening|exacerbation|flare|progress\w*|"
    r"rising|elevated|above goal|inadequately|suboptimal|side effects?|intoleran\w*|decompensat\w*|volume overload)\b",
    re.I,
)
_CHRONIC = re.compile(
    r"\b(?:hypertension|diabetes|hyperlipidemia|hypothyroidism|hashimoto|atrial fibrillation|heart failure|asthma|"
    r"copd|depress\w*|anxiety|bipolar|adhd|psoriasis|atopic dermatitis|eczema|coronary artery disease|cad|insomnia|"
    r"alcohol use disorder|obesity|ckd|chronic kidney|osteoarthritis|migraine|gerd|acne)\b",
    re.I,
)
_UNDIAGNOSED = re.compile(
    r"\b(?:rule out|r/o|concern for|suspicious|workup|work-up|evaluate|evaluation|biopsy|stress test|possible|"
    r"likely|differential|angina|chest pain|palpitations|uncertain)\b",
    re.I,
)
_SYSTEMIC = re.compile(r"\b(?:fever|febrile|systemic|dehydrat\w*|pneumonia|pyelonephritis)\b", re.I)
_ACUTE_UNCOMPLICATED = re.compile(
    r"\b(?:upper respiratory|uri|viral|cold|sinusitis|pharyngitis|strep|otitis|cystitis|uti|sprain|strain|wart|"
    r"verruca|rash|dermatitis|constipation|keratosis|keratoses|nevus|nevi|mole)\b",
    re.I,
)
_TEST = re.compile(
    r"\b(?:labs?|blood work|a1c|cbc|cmp|bmp|lipid|tsh|thyroid|panel|urinalysis|urine|culture|ecg|ekg|echo\w*|"
    r"stress test|x-ray|xray|ultrasound|ct|mri|biopsy|level|liver|kidney function|metabolic|strep test|holter|"
    r"monitor|spirometry|pathology)\b",
    re.I,
)
_INTERPRETATION = re.compile(
    r"\b(?:ecg|ekg|electrocardiogram|x-ray)\b.*\b(?:shows|showed|normal|sinus|interpret)", re.I
)
_RX_ACTION = re.compile(
    r"\b(?:start\w*|begin|prescrib\w*|increase\w*|decrease\w*|reduce\w*|lower\w*|raise\w*|switch\w*|stop\w*|"
    r"discontinu\w*|continue\w*|refill\w*|titrat\w*|change\w*|adjust\w*|taper\w*|hold\w*)\b",
    re.I,
)
_HIGH_RISK = re.compile(
    r"\b(?:hospitali[sz]\w*|admit\w*|emergency surgery|icu|intensive care|do not resuscitate)\b", re.I
)
_OTC = re.compile(
    r"\b(?:over the counter|otc|acetaminophen|tylenol|ibuprofen|advil|motrin|naproxen|aleve|loratadine|cetiriz\w*|"
    r"miralax|polyethylene|saline|sunscreen|moisturiz\w*|vitamin|melatonin|zinc oxide|nicotine)\b",
    re.I,
)


def _level_index(name: str) -> int:
    return LEVELS.index(name)


def estimate(
    statements: Sequence[Statement],
    medications: Sequence[Medication],
    *,
    visit_type: str,
    caregiver_present: bool = False,
    procedures: int = 0,
) -> MDMEstimate:
    assessment = [s.text for s in statements if s.section == "assessment"]
    plan = [s.text for s in statements if s.section == "plan"]
    objective = [s.text for s in statements if s.section == "objective"]
    reasons: list[str] = []

    # ---- problems
    chronic = sum(1 for a in assessment if _CHRONIC.search(a))
    exacerbated = sum(1 for a in assessment if _CHRONIC.search(a) and _EXACERBATION.search(a))
    undiagnosed = sum(1 for a in assessment if _UNDIAGNOSED.search(a))
    systemic = any(_SYSTEMIC.search(a) for a in assessment + objective)
    acute = sum(1 for a in assessment if _ACUTE_UNCOMPLICATED.search(a))
    if exacerbated or chronic >= 2 or undiagnosed or (acute and systemic):
        problems = "moderate"
        reasons.append(
            "problems: chronic illness not at goal" if exacerbated else
            "problems: two or more chronic illnesses" if chronic >= 2 else
            "problems: undiagnosed new problem with uncertain prognosis" if undiagnosed else
            "problems: acute illness with systemic symptoms"
        )  # fmt: skip
    elif chronic or acute >= 2 or (acute and len(assessment) >= 2):
        problems = "low"
        reasons.append("problems: one stable chronic illness or two self-limited problems")
    elif assessment:
        problems = "low" if acute else "straightforward"
        reasons.append(f"problems: {'one acute uncomplicated problem' if acute else 'one minor problem'}")
    else:
        problems = "straightforward"

    # ---- data
    tests = {m.group(0).lower() for t in plan + objective for m in _TEST.finditer(t)}
    interpretation = any(_INTERPRETATION.search(t) for t in objective)
    points = len(tests) + (1 if caregiver_present else 0)
    if interpretation and points >= 3:
        data = "moderate"
        reasons.append("data: tests ordered or reviewed plus an independent interpretation")
    elif points >= 3 or interpretation:
        data = "moderate" if points >= 3 else "low"
        reasons.append(f"data: {points} tests or sources")
    elif points >= 2 or caregiver_present:
        data = "low"
        reasons.append("data: limited (tests or an independent historian)")
    else:
        data = "straightforward"

    # ---- risk
    rx = [m for m in medications if m.status in {"new", "changed", "stopped", "active"}]
    rx_plan = any(_RX_ACTION.search(p) and not _OTC.search(p) for p in plan)
    prescription_drug_management = any(
        not _OTC.search(m.name) and (m.status in {"new", "changed", "stopped"} or rx_plan) for m in rx
    )
    if any(_HIGH_RISK.search(p) for p in plan):
        risk = "high"
        reasons.append("risk: decision about hospitalization")
    elif prescription_drug_management:
        risk = "moderate"
        reasons.append("risk: prescription drug management")
    elif plan or procedures:
        risk = "low"
        reasons.append("risk: over-the-counter drugs, minor procedure or counseling")
    else:
        risk = "straightforward"

    ranks = sorted(_level_index(x) for x in (problems, data, risk))
    level = LEVELS[ranks[1]]  # two of three elements must meet or exceed the level
    table = NEW if visit_type == "new" else ESTABLISHED
    return MDMEstimate(problems=problems, data=data, risk=risk, level=level, code=table[level], reasons=reasons)  # type: ignore[arg-type]


def code_level(code: str) -> str | None:
    for table in (NEW, ESTABLISHED):
        for level, value in table.items():
            if value == code:
                return level
    return None
