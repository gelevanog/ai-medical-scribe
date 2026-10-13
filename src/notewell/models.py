"""Domain models shared by the pipeline, the API, the MCP server and the evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["clinician", "patient", "caregiver", "unknown"]
Section = Literal["subjective", "objective", "assessment", "plan"]
SECTIONS: tuple[Section, ...] = ("subjective", "objective", "assessment", "plan")
Support = Literal["supported", "partial", "unsupported"]
Severity = Literal["error", "warning", "info"]
VisitStatus = Literal["new", "transcribed", "drafted", "signed", "exported"]
FlagKind = Literal[
    "unsupported",
    "partial_support",
    "negation",
    "attribution",
    "hypothetical",
    "dose_range",
    "dose_unit",
    "dose_mismatch",
    "unknown_drug",
    "allergy_conflict",
    "code_invalid",
    "code_not_billable",
    "code_unsupported",
    "code_negated",
    "em_mismatch",
    "em_visit_type",
    "placeholder",
    "readability",
]


class Patient(BaseModel):
    """Registration data from the schedule / EHR. Synthetic in this repository. Never sent to a cloud model."""

    name: str
    dob: str = ""
    sex: Literal["female", "male", "other", "unknown"] = "unknown"
    mrn: str = ""
    phone: str = ""
    address: str = ""
    email: str = ""


class Clinician(BaseModel):
    name: str
    role: str = ""


class Turn(BaseModel):
    idx: int
    speaker: Role
    text: str
    start: float | None = None
    end: float | None = None


class Transcript(BaseModel):
    turns: list[Turn]
    source: Literal["gold", "asr"] = "gold"
    asr_model: str = ""
    duration_s: float = 0.0
    timings: dict[str, float] = Field(default_factory=dict)
    """Seconds per stage: asr, diarization."""
    diarization: dict[str, Any] = Field(default_factory=dict)

    def text(self) -> str:
        return "\n".join(f"[{t.idx}] {t.speaker}: {t.text}" for t in self.turns)


class Evidence(BaseModel):
    turn: int
    quote: str
    verified: bool = False
    score: float = 0.0


class Flag(BaseModel):
    kind: FlagKind
    severity: Severity = "warning"
    message: str
    acknowledged: bool = False
    detail: dict[str, Any] = Field(default_factory=dict)


class Statement(BaseModel):
    id: str
    section: Section
    text: str
    evidence: list[Evidence] = Field(default_factory=list)
    support: Support = "unsupported"
    flags: list[Flag] = Field(default_factory=list)
    origin: Literal["model", "clinician"] = "model"
    edited: bool = False


class Medication(BaseModel):
    name: str
    dose: str = ""
    frequency: str = ""
    route: str = ""
    status: str = "active"
    evidence: list[Evidence] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)


class Allergy(BaseModel):
    substance: str
    reaction: str = ""
    evidence: list[Evidence] = Field(default_factory=list)


class CodeSuggestion(BaseModel):
    system: Literal["ICD-10-CM", "CPT"]
    code: str
    description: str = ""
    rank: int = 1
    rationale: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    valid: bool = False
    billable: bool = False
    flags: list[Flag] = Field(default_factory=list)
    accepted: bool | None = None


class MDMEstimate(BaseModel):
    """Deterministic estimate of the 2021 office E/M level from the note (problems, data, risk)."""

    problems: str
    data: str
    risk: str
    level: Literal["straightforward", "low", "moderate", "high"]
    code: str
    reasons: list[str] = Field(default_factory=list)


def _draft_status() -> dict[str, Literal["draft", "accepted"]]:
    return dict.fromkeys(SECTIONS, "draft")


class Note(BaseModel):
    visit_id: str
    statements: list[Statement] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    after_visit_summary: str = ""
    avs_flags: list[Flag] = Field(default_factory=list)
    icd10: list[CodeSuggestion] = Field(default_factory=list)
    em: CodeSuggestion | None = None
    procedures: list[CodeSuggestion] = Field(default_factory=list)
    mdm: MDMEstimate | None = None
    section_status: dict[str, Literal["draft", "accepted"]] = Field(default_factory=lambda: _draft_status())
    model: str = ""
    created_at: datetime | None = None
    signed_by: str | None = None
    signed_at: datetime | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    phi: dict[str, int] = Field(default_factory=dict)
    """Counts of masked identifiers by type (never the values)."""

    def section(self, name: Section) -> list[Statement]:
        return [s for s in self.statements if s.section == name]

    def open_flags(self) -> list[Flag]:
        flags: list[Flag] = []
        for s in self.statements:
            flags.extend(s.flags)
        for m in self.medications:
            flags.extend(m.flags)
        for c in [*self.icd10, *self.procedures, *([self.em] if self.em else [])]:
            if c.accepted is not False:
                flags.extend(c.flags)
        flags.extend(self.avs_flags)
        return [f for f in flags if f.severity == "error" and not f.acknowledged]


class Visit(BaseModel):
    id: str
    title: str
    specialty: str
    visit_type: Literal["new", "established"] = "established"
    visit_date: str
    patient: Patient
    clinician: Clinician
    status: VisitStatus = "new"
    audio: str | None = None
    """Audio file name under the audio directory (synthetic visits) or the encrypted upload store."""
    transcript: Transcript | None = None
    note: Note | None = None
    synthetic: bool = True
    created_at: datetime | None = None
