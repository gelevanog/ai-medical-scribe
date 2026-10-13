"""Medication sanity checks against Notewell's small open drug list (src/notewell/kb/drugs.yaml).

Catches the errors a scribe makes: a drug that is not in the list (misheard or invented), a unit that cannot be
right (levothyroxine in milligrams), a dose outside the usual range or a daily total over the maximum, a pediatric
weight-based dose outside its mg/kg/day range, a dose that nobody said, a dose that was corrected in the conversation,
and a drug that conflicts with a stated allergy. These are orientation checks for review, not dosing guidance.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from notewell.guards.text import numbers, words_to_digits
from notewell.kb import Drug, drug_aliases, drug_pattern, drugs
from notewell.models import Allergy, Flag, Medication, Turn

_UNIT_ALIASES = {
    "mg": "mg", "milligram": "mg", "milligrams": "mg", "mgs": "mg",
    "mcg": "mcg", "microgram": "mcg", "micrograms": "mcg", "µg": "mcg", "ug": "mcg",
    "g": "g", "gram": "g", "grams": "g",
    "ml": "ml", "milliliter": "ml", "milliliters": "ml", "cc": "ml",
    "unit": "units", "units": "units", "iu": "iu",
    "%": "%", "percent": "%",
    "puff": "puffs", "puffs": "puffs", "spray": "sprays", "sprays": "sprays",
    "tablet": "tablet", "tablets": "tablet", "tab": "tablet", "capsule": "tablet", "capsules": "tablet",
    "patch": "patch", "dose": "dose",
}  # fmt: skip
_DOSE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|mcg|µg|ug|g|ml|cc|units?|iu|%|percent|milligrams?|micrograms?|grams?|milliliters?|"
    r"puffs?|sprays?|tablets?|tabs?|capsules?|patch|dose)\b(?:\s*(?:/|per)\s*(kg|kilogram))?(?:\s*(?:/|per)\s*(day))?",
    re.IGNORECASE,
)
_FREQ = [
    (re.compile(r"\b(?:four times|4 times|qid|q\.i\.d)\b", re.I), 4.0),
    (re.compile(r"\b(?:three times|3 times|tid|t\.i\.d)\b", re.I), 3.0),
    (re.compile(r"\b(?:twice|two times|2 times|bid|b\.i\.d|every 12 hours|q12h?)\b", re.I), 2.0),
    (re.compile(r"\bevery (\d+) hours?\b|\bq(\d+)h\b", re.I), -1.0),
    (re.compile(r"\b(?:weekly|once a week|every week|every other week|every 2 weeks|monthly)\b", re.I), 1 / 7),
    (re.compile(r"\b(?:once|daily|every day|a day|per day|qd|nightly|at bedtime|bedtime|qhs|every morning|in the "
                r"morning|every night|each morning|each night|one time)\b", re.I), 1.0),
]  # fmt: skip
_CORRECTION = re.compile(
    r"\b(?:actually|wait|sorry|I mean|correction|scratch that|no,? (?:it's|it is|wait)|not \d+(?:\.\d+)?|"
    r"they (?:upped|raised|lowered|changed|bumped))\b",
    re.I,
)
_ALLERGY_CLASSES = {
    "penicillin": "penicillin", "penicillins": "penicillin", "amoxicillin": "penicillin", "augmentin": "penicillin",
    "sulfa": "sulfonamide", "sulfa drugs": "sulfonamide", "bactrim": "sulfonamide", "sulfonamide": "sulfonamide",
    "nsaid": "nsaid", "nsaids": "nsaid", "ibuprofen": "nsaid", "aspirin": "nsaid", "naproxen": "nsaid",
    "codeine": "opioid", "morphine": "opioid", "opioids": "opioid", "hydrocodone": "opioid", "tramadol": "opioid",
    "cephalosporins": "cephalosporin", "cephalexin": "cephalosporin", "keflex": "cephalosporin",
    "tetracycline": "tetracycline", "doxycycline": "tetracycline", "macrolides": "macrolide",
    "azithromycin": "macrolide", "erythromycin": "macrolide", "fluoroquinolones": "fluoroquinolone",
    "ciprofloxacin": "fluoroquinolone", "statins": "statin", "ace inhibitors": "ace_inhibitor",
    "lisinopril": "ace_inhibitor", "latex": "latex", "lamotrigine": "anticonvulsant",
}  # fmt: skip


def normalize_drug(name: str) -> Drug | None:
    aliases = drug_aliases()
    lowered = name.lower().strip()
    if lowered in aliases:
        return drugs()[aliases[lowered]]
    match = drug_pattern().search(lowered)
    if match:
        return drugs()[aliases[match.group(1).lower()]]
    return None


def parse_dose(text: str) -> tuple[float, str, bool] | None:
    """(amount, unit, per_kg_per_day) from "20 mg", "88 micrograms", "90 mg/kg/day"."""
    match = _DOSE.search(words_to_digits(text))
    if not match:
        return None
    unit = _UNIT_ALIASES.get(match.group(2).lower(), match.group(2).lower())
    return float(match.group(1)), unit, bool(match.group(3))


def times_per_day(frequency: str) -> float | None:
    for pattern, value in _FREQ:
        m = pattern.search(frequency)
        if m:
            if value < 0:
                hours = next(g for g in m.groups() if g)
                return 24 / float(hours) if float(hours) > 0 else None
            return value
    return None


def weight_kg(turns: Sequence[Turn]) -> float | None:
    for turn in turns:
        text = words_to_digits(turn.text)
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:kilograms?|kilos?|kg)\b", text, re.I)
        if m and re.search(r"weigh|weight|kilo", text, re.I):
            return float(m.group(1))
    return None


def _num(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def check_medication(
    med: Medication,
    *,
    turns: Sequence[Turn],
    allergies: Sequence[Allergy],
    pediatric: bool = False,
) -> list[Flag]:
    flags: list[Flag] = []
    drug = normalize_drug(med.name)
    if drug is None:
        flags.append(
            Flag(
                kind="unknown_drug",
                severity="warning",
                message=f"{med.name!r} is not in the drug list: check the spelling and the dose yourself.",
            )
        )
    dose = parse_dose(f"{med.dose} {med.frequency}") if med.dose else None
    if drug is not None and dose is not None:
        amount, unit, per_kg = dose
        unit_ok = (
            unit in drug.units
            or (unit in {"ml", "tablet", "dose", "patch"})
            or (unit == "iu" and "units" in drug.units)
        )
        if not unit_ok:
            flags.append(
                Flag(
                    kind="dose_unit",
                    severity="error",
                    message=f"{drug.name} is dosed in {', '.join(drug.units)}, not {unit}.",
                    detail={"unit": unit, "expected": list(drug.units)},
                )
            )
        elif unit == drug.units[0] and not per_kg:
            low, high = drug.dose
            per_day = times_per_day(med.frequency or "")
            if amount < low / 2 or amount > high * 1.5:
                flags.append(
                    Flag(
                        kind="dose_range",
                        severity="error",
                        message=f"{_num(amount)} {unit} is outside the usual {_num(low)}-{_num(high)} {unit} per dose "
                        f"for {drug.name}.",
                        detail={"amount": amount, "range": [low, high]},
                    )
                )
            elif per_day and amount * per_day > drug.max_daily * 1.05:
                flags.append(
                    Flag(
                        kind="dose_range",
                        severity="error",
                        message=f"{_num(amount * per_day)} {unit} per day exceeds the usual maximum of "
                        f"{_num(drug.max_daily)} {unit} for {drug.name}.",
                        detail={"daily": amount * per_day, "max_daily": drug.max_daily},
                    )
                )
        if pediatric and unit == "mg":
            flags.extend(_pediatric(drug, amount, per_kg, med, turns))
    if dose is not None:
        flags.extend(_dose_in_transcript(med, dose[0], turns, drug))
    flags.extend(_allergy(med, drug, allergies))
    return flags


def _pediatric(drug: Drug, amount: float, per_kg: bool, med: Medication, turns: Sequence[Turn]) -> list[Flag]:
    kg = weight_kg(turns)
    if kg is None or kg <= 0:
        return []
    per_day = times_per_day(med.frequency or "") or 1.0
    mg_kg_day = amount if per_kg else amount * per_day / kg
    if drug.peds_mg_kg_day:
        low, high = drug.peds_mg_kg_day
    elif drug.peds_mg_kg_dose and drug.peds_max_mg_kg_day:
        low, high = drug.peds_mg_kg_dose[0] * 1, drug.peds_max_mg_kg_day
    else:
        return []
    if mg_kg_day < low * 0.8 or mg_kg_day > high * 1.1:
        return [
            Flag(
                kind="dose_range",
                severity="error",
                message=f"{drug.name}: {mg_kg_day:.0f} mg/kg/day at {kg:g} kg is outside the usual pediatric "
                f"{_num(low)}-{_num(high)} mg/kg/day.",
                detail={"mg_kg_day": round(mg_kg_day, 1), "weight_kg": kg},
            )
        ]
    return []


def _dose_in_transcript(med: Medication, amount: float, turns: Sequence[Turn], drug: Drug | None) -> list[Flag]:
    """The dose must have been said near the drug's name; a dose with a correction gets a confirmation prompt."""
    names = {med.name.lower()}
    if drug is not None:
        names |= {drug.name, *drug.brands}
    mention_turns = [
        i for i, t in enumerate(turns) if any(re.search(rf"\b{re.escape(n)}\b", t.text, re.I) for n in names)
    ]
    if not mention_turns:
        return []
    nearby: set[str] = set()
    window_texts: list[str] = []
    for i in mention_turns:
        for j in range(max(0, i - 1), min(len(turns), i + 3)):
            nearby |= numbers(turns[j].text)
            window_texts.append(turns[j].text)
    value = _num(amount)
    flags: list[Flag] = []
    if value not in nearby:
        flags.append(
            Flag(
                kind="dose_mismatch",
                severity="error",
                message=f"The dose {value} for {med.name} was not said near the drug's name in the transcript.",
                detail={"dose": value},
            )
        )
    for i in mention_turns:
        text = words_to_digits(turns[i].text)
        windows = [
            text[m.start() : m.start() + 140] for n in names for m in re.finditer(rf"\b{re.escape(n)}\b", text, re.I)
        ]
        if i + 1 < len(turns) and turns[i + 1].speaker != turns[i].speaker:
            windows.append(words_to_digits(turns[i + 1].text)[:160])  # the answer to "are you taking the X?"
        window = next(
            (w for w in windows if len(set(re.findall(r"\d+(?:\.\d+)?", w))) > 1 and _CORRECTION.search(w)), ""
        )
        cue = _CORRECTION.search(window)
        limit = (cue.end() + 16) if cue else len(window)
        doses = {
            m.group(1)
            for m in re.finditer(
                r"(\d+(?:\.\d+)?)(?=\s*(?:mg|mcg|ml|milligrams?|micrograms?|milliliters?|units?|puffs?|now|\b(?!\s*(?:to|or|hours?|times|days?|weeks?))))",
                window,
            )
            if m.start() < limit
        }
        if window:
            flags.append(
                Flag(
                    kind="dose_mismatch",
                    severity="warning",
                    message=f"The {med.name} dose was corrected in the conversation ({', '.join(sorted(doses))}); "
                    "confirm the final dose.",
                    detail={"doses": sorted(doses), "turn": turns[i].idx},
                )
            )
            break
    return flags


def _allergy(med: Medication, drug: Drug | None, allergies: Sequence[Allergy]) -> list[Flag]:
    if drug is None or med.status == "stopped":
        return []
    flags: list[Flag] = []
    for allergy in allergies:
        substance = allergy.substance.lower().strip()
        allergic_class = _ALLERGY_CLASSES.get(substance)
        allergic_drug = normalize_drug(substance)
        conflict = (allergic_drug is not None and allergic_drug.name == drug.name) or (
            allergic_class is not None and allergic_class in drug.classes
        )
        if conflict:
            flags.append(
                Flag(
                    kind="allergy_conflict",
                    severity="error",
                    message=f"{drug.name} conflicts with the documented {allergy.substance} allergy"
                    + (f" ({allergy.reaction})" if allergy.reaction else "")
                    + ".",
                )
            )
        elif allergic_class == "penicillin" and "cephalosporin" in drug.classes:
            flags.append(
                Flag(
                    kind="allergy_conflict",
                    severity="warning",
                    message=f"{drug.name} is a cephalosporin and the patient reports a penicillin allergy"
                    + (f" ({allergy.reaction})" if allergy.reaction else "")
                    + ": low cross-reactivity, confirm it was considered.",
                )
            )
    return flags
