"""Small curated knowledge files: clinical concepts (negation and grounding), the drug list and the CPT list."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml


def _yaml(name: str) -> Any:
    return yaml.safe_load(resources.files("notewell.kb").joinpath(name).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Drug:
    name: str
    brands: tuple[str, ...]
    classes: tuple[str, ...]
    units: tuple[str, ...]
    routes: tuple[str, ...]
    dose: tuple[float, float]
    max_daily: float
    peds_mg_kg_day: tuple[float, float] | None = None
    peds_mg_kg_dose: tuple[float, float] | None = None
    peds_max_mg_kg_day: float | None = None


@lru_cache(maxsize=1)
def drugs() -> dict[str, Drug]:
    out: dict[str, Drug] = {}
    for row in _yaml("drugs.yaml"):
        drug = Drug(
            name=row["name"],
            brands=tuple(row.get("brands") or ()),
            classes=tuple(row.get("classes") or ()),
            units=tuple(str(u) for u in row.get("units") or ()),
            routes=tuple(row.get("routes") or ()),
            dose=(float(row["dose"][0]), float(row["dose"][1])),
            max_daily=float(row["max_daily"]),
            peds_mg_kg_day=tuple(row["peds_mg_kg_day"]) if row.get("peds_mg_kg_day") else None,  # type: ignore[arg-type]
            peds_mg_kg_dose=tuple(row["peds_mg_kg_dose"]) if row.get("peds_mg_kg_dose") else None,  # type: ignore[arg-type]
            peds_max_mg_kg_day=row.get("peds_max_mg_kg_day"),
        )
        out[drug.name] = drug
    return out


@lru_cache(maxsize=1)
def drug_aliases() -> dict[str, str]:
    """Every generic and brand spelling (lowercase) -> generic name. Longer aliases are matched first by callers."""
    aliases: dict[str, str] = {}
    for drug in drugs().values():
        aliases[drug.name] = drug.name
        for brand in drug.brands:
            aliases.setdefault(brand, drug.name)
    return aliases


@lru_cache(maxsize=1)
def drug_pattern() -> re.Pattern[str]:
    names = sorted(drug_aliases(), key=len, reverse=True)
    return re.compile(r"(?<![\w-])(" + "|".join(re.escape(n) for n in names) + r")(?![\w-])", re.IGNORECASE)


@lru_cache(maxsize=1)
def concepts() -> dict[str, tuple[str, ...]]:
    return {key: tuple(forms) for key, forms in _yaml("concepts.yaml").items()}


def _form_regex(form: str) -> str:
    if form.endswith("*"):
        return r"\b" + re.escape(form[:-1]) + r"[\w'-]*"
    return r"\b" + re.escape(form) + r"\b"


@lru_cache(maxsize=1)
def concept_patterns() -> dict[str, re.Pattern[str]]:
    return {
        key: re.compile("|".join(_form_regex(f) for f in sorted(forms, key=len, reverse=True)), re.IGNORECASE)
        for key, forms in concepts().items()
    }


def find_concepts(text: str) -> list[tuple[str, int, int]]:
    """(concept, start, end) for every clinical concept mention, longest-first without overlaps."""
    hits: list[tuple[str, int, int]] = []
    for key, pattern in concept_patterns().items():
        hits.extend((key, m.start(), m.end()) for m in pattern.finditer(text))
    hits.sort(key=lambda h: (h[1], -(h[2] - h[1])))
    chosen: list[tuple[str, int, int]] = []
    for hit in hits:
        if chosen and hit[1] < chosen[-1][2]:
            continue
        chosen.append(hit)
    return chosen


@lru_cache(maxsize=1)
def cpt() -> dict[str, dict[str, Any]]:
    data = _yaml("cpt.yaml")
    out: dict[str, dict[str, Any]] = {}
    for group in ("em_new", "em_established"):
        for code, row in data[group].items():
            out[str(code)] = {"group": group, "summary": row["summary"], "mdm": row["mdm"]}
    for group in ("psychiatry", "procedures"):
        for code, summary in data[group].items():
            out[str(code)] = {"group": group, "summary": summary}
    return out


_EXTRA_CLINICAL = """
a1c ldl hdl tsh bmi phq gad gad-7 phq-9 audit-c ekg ecg echo echocardiogram mri ct x-ray ultrasound covid strep flu
tdap mmr dtap ipv hpv rsv hep hepatitis cbc cmp bmp ua inr psa egfr afib a-fib copd chf hfref hfpef dvt pe uti
uri gerd adhd ocd ptsd ibs ra lupus hiv std sti otc iud ppd tb bp hr rr spo2 o2 icd cpt fhir soap nsaid nsaids
ssri ssris snri acei arb statin statins zpack z-pack epipen ace bpm mmhg celsius fahrenheit mg mcg ml kg lbs
apgar denver vanderbilt snellen tanner gleason mallampati babinski romberg tinel phalen lachman mcmurray
spurling murphy mcburney rovsing homan homans kernig brudzinski epley valsalva holter zio kardia apple watch fitbit
hashimoto hashimoto's graves graves' crohn crohn's parkinson parkinson's alzheimer alzheimer's raynaud raynaud's
sjogren sjogren's addison addison's cushing cushing's kawasaki osgood schlatter tourette tourette's asperger bell's
lyme hodgkin hodgkin's barrett barrett's wolff white reiter's guillain barre meniere meniere's huntington marfan
kaposi epstein barr wilms paget tay sachs turner's klinefelter heberden bouchard baker's achilles morton's dupuytren
pfizer moderna novavax johnson medicare medicaid tricare aetna cigna humana kaiser blue cross shield united healthcare
""".split()


@lru_cache(maxsize=1)
def clinical_vocabulary() -> frozenset[str]:
    words: set[str] = set(_EXTRA_CLINICAL)
    for alias in drug_aliases():
        words.update(re.findall(r"[a-z][a-z0-9'-]+", alias))
    for forms in concepts().values():
        for form in forms:
            words.update(re.findall(r"[a-z][a-z'-]+", form.rstrip("*")))
    return frozenset(words)
