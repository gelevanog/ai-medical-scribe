"""FHIR R4 export of a signed note: a transaction Bundle with Patient, Practitioner, Encounter, Composition (the SOAP
sections), DocumentReference (the PDF and the text), Condition (accepted ICD-10-CM codes), MedicationStatement and
AllergyIntolerance, plus a Provenance that records that an AI drafted the note and a clinician signed it.

Resources use urn:uuid references inside the bundle, so a FHIR server (HAPI FHIR or the built-in mock) assigns ids.
"""

from __future__ import annotations

import base64
import html
import uuid
from datetime import UTC, datetime
from typing import Any

from notewell.models import SECTIONS, Note, Visit

LOINC = "http://loinc.org"
ICD10CM = "http://hl7.org/fhir/sid/icd-10-cm"
CPT_SYSTEM = "http://www.ama-assn.org/go/cpt"
SECTION_LOINC = {
    "subjective": ("61150-9", "Subjective Narrative"),
    "objective": ("61149-1", "Objective Narrative"),
    "assessment": ("51848-0", "Evaluation note"),
    "plan": ("18776-5", "Plan of care note"),
}
NOTE_TYPE = ("11506-3", "Progress note")
AI_EXTENSION = "https://github.com/gelevanog/ai-medical-scribe/fhir/StructureDefinition/ai-drafted"


def _uuid(seed: str) -> str:
    return f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, seed)}"


def _div(paragraphs: list[str]) -> str:
    body = "".join(f"<p>{html.escape(p)}</p>" for p in paragraphs) or "<p>(none)</p>"
    return f'<div xmlns="http://www.w3.org/1999/xhtml">{body}</div>'


def _name(full: str) -> dict[str, Any]:
    parts = full.replace("Dr. ", "").split()
    return {"text": full, "family": parts[-1] if parts else full, "given": parts[:-1] or parts}


def note_text(note: Note) -> str:
    lines: list[str] = []
    for section in SECTIONS:
        lines.append(section.upper())
        lines.extend(f"- {s.text}" for s in note.section(section))
        if section == "subjective":
            if note.medications:
                lines.append(
                    "Medications: "
                    + "; ".join(f"{m.name} {m.dose} {m.frequency} ({m.status})".strip() for m in note.medications)
                )
            lines.append(
                "Allergies: "
                + (
                    "; ".join(f"{a.substance} ({a.reaction})" if a.reaction else a.substance for a in note.allergies)
                    or "none reported"
                )
            )
        lines.append("")
    return "\n".join(lines).strip()


def build_bundle(
    visit: Visit, note: Note, *, pdf: bytes | None = None, organization: str = "Notewell Demo Clinic"
) -> dict[str, Any]:
    if note.signed_at is None or not note.signed_by:
        raise ValueError("only a signed note can be exported")
    now = datetime.now(UTC).isoformat(timespec="seconds")
    signed = note.signed_at.isoformat(timespec="seconds")
    patient_ref = _uuid(f"patient:{visit.id}")
    practitioner_ref = _uuid(f"practitioner:{visit.clinician.name}")
    encounter_ref = _uuid(f"encounter:{visit.id}")
    composition_ref = _uuid(f"composition:{visit.id}")
    docref_ref = _uuid(f"docref:{visit.id}")
    org_ref = _uuid(f"organization:{organization}")
    entries: list[dict[str, Any]] = []

    def add(ref: str, resource: dict[str, Any], *, conditional: str | None = None) -> None:
        request: dict[str, Any] = {"method": "POST", "url": resource["resourceType"]}
        if conditional:
            request["ifNoneExist"] = conditional
        entries.append({"fullUrl": ref, "resource": resource, "request": request})

    p = visit.patient
    patient: dict[str, Any] = {
        "resourceType": "Patient",
        "identifier": [{"system": "urn:notewell:demo:mrn", "value": p.mrn}] if p.mrn else [],
        "name": [_name(p.name)],
        "gender": p.sex if p.sex in {"female", "male", "other", "unknown"} else "unknown",
        "meta": {"tag": [{"system": "urn:notewell", "code": "synthetic", "display": "Synthetic demo data"}]},
    }
    if p.dob:
        patient["birthDate"] = p.dob
    if p.phone:
        patient["telecom"] = [{"system": "phone", "value": p.phone}]
    if p.address:
        patient["address"] = [{"text": p.address}]
    add(patient_ref, patient, conditional=f"identifier=urn:notewell:demo:mrn|{p.mrn}" if p.mrn else None)
    add(org_ref, {"resourceType": "Organization", "name": organization})
    add(practitioner_ref, {"resourceType": "Practitioner", "name": [_name(visit.clinician.name)]})
    add(
        encounter_ref,
        {
            "resourceType": "Encounter",
            "status": "finished",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": "AMB",
                "display": "ambulatory",
            },
            "type": [{"coding": [{"system": CPT_SYSTEM, "code": note.em.code}]}]
            if note.em and note.em.accepted is not False
            else [],
            "subject": {"reference": patient_ref},
            "participant": [{"individual": {"reference": practitioner_ref}}],
            "period": {"start": visit.visit_date, "end": visit.visit_date},
            "serviceProvider": {"reference": org_ref},
        },
    )
    condition_refs: list[str] = []
    for code in note.icd10:
        if code.accepted is not True or not code.valid:
            continue
        ref = _uuid(f"condition:{visit.id}:{code.code}")
        condition_refs.append(ref)
        add(
            ref,
            {
                "resourceType": "Condition",
                "clinicalStatus": {
                    "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]
                },
                "verificationStatus": {
                    "coding": [
                        {"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}
                    ]
                },
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                                "code": "encounter-diagnosis",
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [{"system": ICD10CM, "code": code.code, "display": code.description}],
                    "text": code.description,
                },
                "subject": {"reference": patient_ref},
                "encounter": {"reference": encounter_ref},
                "recordedDate": signed,
                "recorder": {"reference": practitioner_ref},
            },
        )
    for i, med in enumerate(note.medications):
        status = {"stopped": "stopped", "new": "intended"}.get(med.status, "active")
        add(
            _uuid(f"medstatement:{visit.id}:{i}"),
            {
                "resourceType": "MedicationStatement",
                "status": status,
                "medicationCodeableConcept": {"text": med.name},
                "subject": {"reference": patient_ref},
                "context": {"reference": encounter_ref},
                "dosage": [
                    {"text": " ".join(x for x in (med.dose, med.frequency, med.route) if x).strip() or "as directed"}
                ],
            },
        )
    for i, allergy in enumerate(note.allergies):
        resource: dict[str, Any] = {
            "resourceType": "AllergyIntolerance",
            "clinicalStatus": {
                "coding": [
                    {"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", "code": "active"}
                ]
            },
            "code": {"text": allergy.substance},
            "patient": {"reference": patient_ref},
        }
        if allergy.reaction:
            resource["reaction"] = [{"manifestation": [{"text": allergy.reaction}]}]
        add(_uuid(f"allergy:{visit.id}:{i}"), resource)

    sections = []
    for section in SECTIONS:
        code, display = SECTION_LOINC[section]
        texts = [s.text for s in note.section(section)]
        entry: dict[str, Any] = {
            "title": section.capitalize(),
            "code": {"coding": [{"system": LOINC, "code": code, "display": display}]},
            "text": {"status": "generated", "div": _div(texts)},
        }
        if section == "assessment" and condition_refs:
            entry["entry"] = [{"reference": r} for r in condition_refs]
        sections.append(entry)
    if note.after_visit_summary:
        sections.append(
            {
                "title": "After-visit summary",
                "code": {"coding": [{"system": LOINC, "code": "34109-9", "display": "Note"}]},
                "text": {"status": "generated", "div": _div([note.after_visit_summary])},
            }
        )
    add(
        composition_ref,
        {
            "resourceType": "Composition",
            "status": "final",
            "type": {"coding": [{"system": LOINC, "code": NOTE_TYPE[0], "display": NOTE_TYPE[1]}]},
            "subject": {"reference": patient_ref},
            "encounter": {"reference": encounter_ref},
            "date": signed,
            "author": [{"reference": practitioner_ref}],
            "title": f"{visit.title} - progress note",
            "attester": [{"mode": "legal", "time": signed, "party": {"reference": practitioner_ref}}],
            "custodian": {"reference": org_ref},
            "extension": [
                {"url": AI_EXTENSION, "valueString": f"Drafted by {note.model}; reviewed and signed by the clinician."}
            ],
            "section": sections,
        },
    )
    text = note_text(note)
    content = [
        {
            "attachment": {
                "contentType": "text/plain",
                "data": base64.b64encode(text.encode()).decode(),
                "title": "SOAP note (text)",
            }
        }
    ]
    if pdf is not None:
        content.insert(
            0,
            {
                "attachment": {
                    "contentType": "application/pdf",
                    "data": base64.b64encode(pdf).decode(),
                    "title": "SOAP note (PDF)",
                }
            },
        )
    add(
        docref_ref,
        {
            "resourceType": "DocumentReference",
            "status": "current",
            "docStatus": "final",
            "type": {"coding": [{"system": LOINC, "code": NOTE_TYPE[0], "display": NOTE_TYPE[1]}]},
            "subject": {"reference": patient_ref},
            "date": now,
            "author": [{"reference": practitioner_ref}],
            "authenticator": {"reference": practitioner_ref},
            "content": content,
            "context": {"encounter": [{"reference": encounter_ref}], "related": [{"reference": composition_ref}]},
        },
    )
    add(
        _uuid(f"provenance:{visit.id}"),
        {
            "resourceType": "Provenance",
            "target": [{"reference": composition_ref}, {"reference": docref_ref}],
            "recorded": now,
            "activity": {"text": "AI-drafted, clinician-reviewed and signed"},
            "agent": [
                {"type": {"text": "author (AI draft)"}, "who": {"display": note.model or "Notewell"}},
                {"type": {"text": "attester"}, "who": {"reference": practitioner_ref}},
            ],
        },
    )
    return {"resourceType": "Bundle", "type": "transaction", "timestamp": now, "entry": entries}


def validate_bundle(bundle: dict[str, Any]) -> list[str]:
    """Structural validation of every resource with the fhir.resources (R4B) models; returns the errors."""
    from importlib import import_module

    errors: list[str] = []
    for entry in bundle.get("entry", []):
        resource = dict(entry["resource"])
        rtype = resource.get("resourceType", "")
        try:
            module = import_module(f"fhir.resources.R4B.{rtype.lower()}")
            getattr(module, rtype).model_validate(resource)
        except Exception as exc:  # pydantic ValidationError or an unknown resource type
            errors.append(f"{rtype}: {str(exc).splitlines()[0][:300]}")
    return errors
