"""The synthetic visit dataset: dialogues with gold SOAP facts, medications, codes, identifiers and traps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from notewell.models import Clinician, Patient, Transcript, Turn, Visit

SPECIALTIES = ("primary_care", "cardiology", "dermatology", "pediatrics", "mental_health")
SPECIALTY_LABELS = {
    "primary_care": "Primary care",
    "cardiology": "Cardiology",
    "dermatology": "Dermatology",
    "pediatrics": "Pediatrics",
    "mental_health": "Mental health",
}


class GoldFact(BaseModel):
    id: str
    section: Literal["subjective", "objective", "assessment", "plan"]
    text: str
    polarity: Literal["present", "absent"] = "present"
    evidence: str
    match: list[list[str]]


class GoldMedication(BaseModel):
    name: str
    dose: str = ""
    frequency: str = ""
    route: str = ""
    status: str = "active"
    evidence: str = ""


class GoldDiagnosis(BaseModel):
    code: str
    description: str = ""
    primary: bool = False
    evidence: str = ""
    alternatives: list[str] = Field(default_factory=list)


class GoldPHI(BaseModel):
    type: str
    value: str
    who: str = "other"


class GoldTrap(BaseModel):
    type: str
    text: str
    evidence: str
    match: list[list[str]]


class Gold(BaseModel):
    facts: list[GoldFact]
    medications: list[GoldMedication] = Field(default_factory=list)
    allergies: list[dict[str, Any]] = Field(default_factory=list)
    diagnoses: list[GoldDiagnosis] = Field(default_factory=list)
    em_code: str = ""
    em_rationale: str = ""
    procedures: list[dict[str, Any]] = Field(default_factory=list)
    phi: list[GoldPHI] = Field(default_factory=list)

    @property
    def primary(self) -> GoldDiagnosis:
        return next(d for d in self.diagnoses if d.primary)


class Case(BaseModel):
    """One synthetic visit with its gold annotations."""

    id: str
    specialty: str
    title: str
    visit_type: Literal["new", "established"]
    visit_date: str
    patient: Patient
    clinician: Clinician
    speakers: dict[str, dict[str, str]]
    dialogue: list[Turn]
    gold: Gold
    traps: list[GoldTrap] = Field(default_factory=list)

    def transcript(self) -> Transcript:
        return Transcript(turns=[t.model_copy() for t in self.dialogue], source="gold")

    def visit(self, *, audio: str | None = None) -> Visit:
        return Visit(
            id=self.id,
            title=self.title,
            specialty=self.specialty,
            visit_type=self.visit_type,
            visit_date=self.visit_date,
            patient=self.patient,
            clinician=self.clinician,
            audio=audio,
            synthetic=True,
        )


def load_case(path: Path) -> Case:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["dialogue"] = [
        {"idx": i, "speaker": t["speaker"], "text": str(t["text"]).strip()} for i, t in enumerate(raw["dialogue"])
    ]
    patient = dict(raw["patient"])
    for key in ("dob", "phone", "mrn", "address", "email"):
        if key in patient and patient[key] is not None:
            patient[key] = str(patient[key])
    patient = {k: v for k, v in patient.items() if k in Patient.model_fields}
    raw["patient"] = patient
    raw["visit_date"] = str(raw["visit_date"])
    gold = raw["gold"]
    gold["em_code"] = str(gold.get("em_code", ""))
    for d in gold.get("diagnoses", []):
        d["code"] = str(d["code"])
        d["alternatives"] = [str(a) for a in d.get("alternatives") or []]
    for p in gold.get("procedures") or []:
        p["code"] = str(p["code"])
    for m in gold.get("medications", []):
        for key in ("dose", "frequency", "route", "status", "evidence"):
            if m.get(key) is not None:
                m[key] = str(m[key])
    for p in gold.get("phi", []):
        p["value"] = str(p["value"])
    gold["procedures"] = gold.get("procedures") or []
    raw["traps"] = raw.get("traps") or []
    return Case.model_validate(raw)


def load_cases(directory: Path, *, ids: list[str] | None = None) -> list[Case]:
    cases = [load_case(p) for p in sorted(directory.rglob("*.yaml"))]
    order = {s: i for i, s in enumerate(SPECIALTIES)}
    cases.sort(key=lambda c: (order.get(c.specialty, 99), c.id))
    if ids:
        wanted = set(ids)
        cases = [c for c in cases if c.id in wanted]
    return cases
