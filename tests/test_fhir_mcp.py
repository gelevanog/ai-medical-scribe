from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from notewell.fhir.export import build_bundle, note_text, validate_bundle
from notewell.fhir.mock import MockFHIRServer
from notewell.mcp_server import NotewellTools, build_server
from notewell.models import SECTIONS
from notewell.pdf import render_pdf
from notewell.services import Services


def _signed(services: Services, vid: str = "pc-01"):  # type: ignore[no-untyped-def]
    visit = services.draft(vid, "Dr. Test")
    note = visit.note
    assert note is not None
    for c in note.icd10[:1]:
        c.accepted = True
    note.signed_by, note.signed_at = "Dr. Test", datetime.now(UTC)
    return visit, note


def test_bundle_is_valid_fhir_and_complete(services: Services) -> None:
    visit, note = _signed(services)
    bundle = build_bundle(visit, note, pdf=render_pdf(visit, note))
    assert validate_bundle(bundle) == []
    types = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert {
        "Patient",
        "Practitioner",
        "Encounter",
        "Composition",
        "DocumentReference",
        "Condition",
        "Provenance",
    } <= set(types)
    composition = next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Composition")
    assert [s["title"] for s in composition["section"]][:4] == [s.capitalize() for s in SECTIONS]
    condition = next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Condition")
    assert condition["code"]["coding"][0]["system"] == "http://hl7.org/fhir/sid/icd-10-cm"
    docref = next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "DocumentReference")
    assert docref["content"][0]["attachment"]["contentType"] == "application/pdf"
    assert "SUBJECTIVE" in note_text(note)


def test_unsigned_note_cannot_be_exported(services: Services) -> None:
    visit = services.draft("pc-02", "Dr. Test")
    with pytest.raises(ValueError, match="signed"):
        build_bundle(visit, visit.note)  # type: ignore[arg-type]


def test_mock_server_resolves_references_and_conditional_create(services: Services) -> None:
    visit, note = _signed(services)
    server = MockFHIRServer()
    bundle = build_bundle(visit, note)
    first = server.transaction(bundle)
    second = server.transaction(build_bundle(visit, note))
    assert first["entry"][0]["response"]["status"] == "201 Created"
    assert second["entry"][0]["response"]["status"] == "200 OK"  # the Patient already exists (ifNoneExist)
    patient_ref = first["entry"][0]["response"]["location"].split("/_history")[0]
    conditions = server.search("Condition", {"subject": patient_ref})
    assert conditions["total"] >= 1
    assert "urn:uuid" not in json.dumps(server.resources)
    with pytest.raises(ValueError):
        server.transaction({"resourceType": "Patient"})


def test_mcp_tools_never_return_identifiers(services: Services) -> None:
    tools = NotewellTools(services.settings, services=services)
    visit = services.visit("pc-01")
    secrets = [visit.patient.name, *visit.patient.name.split(), visit.patient.phone, visit.patient.mrn]
    outputs = [
        tools.list_visits(),
        tools.get_transcript("pc-01"),
        tools.draft_note("pc-01"),
        tools.suggest_codes("pc-01"),
    ]
    dump = json.dumps(outputs)
    for secret in secrets:
        if secret:
            assert secret not in dump, secret
    assert "[PATIENT]" in dump


def test_mcp_check_note_flags_negation_and_grounds_placeholders(services: Services) -> None:
    tools = NotewellTools(services.settings, services=services)
    transcript = tools.get_transcript("pc-01")
    assert transcript["turns"]
    result = tools.check_note(
        "pc-01",
        [
            {"section": "subjective", "text": "Reports chest pain"},
            {"section": "objective", "text": "Blood pressure 152 over 94"},
            {"section": "subjective", "text": "Patient has a broken arm"},
        ],
    )
    by = result["results"]
    assert any("negation" in f for f in by[0]["flags"])
    assert by[1]["support"] == "supported" and not by[1]["flags"]
    assert by[2]["support"] == "unsupported"


def test_mcp_export_refused_before_signature(services: Services) -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    tools = NotewellTools(services.settings, services=services)
    tools.draft_note("pc-03")
    with pytest.raises(ToolError, match="signed"):
        tools.export_fhir("pc-03")


def test_mcp_validate_and_search(services: Services) -> None:
    tools = NotewellTools(services.settings, services=services)
    statuses = {c["code"]: c["status"] for c in tools.validate_codes(["I10", "E11", "E99.99"])["codes"]}
    assert statuses == {"I10": "valid", "E11": "not_billable", "E99.99": "invalid"}
    assert tools.search_icd10("essential hypertension")["results"][0]["code"] == "I10"


def test_mcp_server_lists_tools(services: Services) -> None:
    import asyncio

    server = build_server(NotewellTools(services.settings, services=services))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"transcribe_visit", "draft_note", "suggest_codes", "export_fhir", "check_note", "get_transcript"} <= names
    assert "sign_note" not in names
