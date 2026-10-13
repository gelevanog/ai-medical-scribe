from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from notewell.api.app import create_app
from notewell.config import Settings
from notewell.llm.fake import FakeScribe
from notewell.llm.resilient import ResilientModel
from notewell.services import Services, build_services


@pytest.fixture
def client(settings: Settings, services: Services) -> TestClient:
    return TestClient(create_app(settings, services))


def _draft(client: TestClient, vid: str = "pc-01") -> dict:  # type: ignore[type-arg]
    r = client.post(f"/api/visits/{vid}/draft")
    assert r.status_code == 200, r.text
    return r.json()


def _resolve_everything(client: TestClient, vid: str, visit: dict) -> dict:  # type: ignore[type-arg]
    note = visit["note"]
    for s in note["statements"]:
        for f in s["flags"]:
            if f["severity"] == "error":
                visit = client.post(
                    f"/api/visits/{vid}/acknowledge", json={"target": s["id"], "kind": f["kind"]}
                ).json()
    for i, m in enumerate(visit["note"]["medications"]):
        for f in m["flags"]:
            if f["severity"] == "error":
                visit = client.post(
                    f"/api/visits/{vid}/acknowledge", json={"target": f"med:{i}", "kind": f["kind"]}
                ).json()
    for c in [*visit["note"]["icd10"], *visit["note"]["procedures"], visit["note"]["em"]]:
        for f in c["flags"]:
            if f["severity"] == "error":
                visit = client.post(
                    f"/api/visits/{vid}/acknowledge", json={"target": f"code:{c['code']}", "kind": f["kind"]}
                ).json()
    for f in visit["note"]["avs_flags"]:
        if f["severity"] == "error":
            visit = client.post(f"/api/visits/{vid}/acknowledge", json={"target": "avs", "kind": f["kind"]}).json()
    return visit


def test_health_and_visit_list(client: TestClient) -> None:
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["llm"] == "fake/fake-scribe" and health["demo_mode"]
    visits = client.get("/api/visits").json()
    assert len(visits) >= 30 and {v["status"] for v in visits} == {"transcribed"}
    one = client.get("/api/visits/pc-01").json()
    assert one["patient"]["name"] and len(one["transcript"]["turns"]) > 20
    assert client.get("/api/visits/nope").status_code == 404


def test_masked_view_has_no_identifiers(client: TestClient) -> None:
    visit = client.get("/api/visits/pc-01").json()
    masked = client.get("/api/visits/pc-01/masked").json()
    text = json.dumps(masked)
    assert visit["patient"]["name"] not in text and "[PATIENT]" in text
    assert masked["counts"]["NAME"] >= 1


def test_draft_review_sign_export_flow(client: TestClient) -> None:
    visit = _draft(client)
    assert visit["status"] == "drafted" and visit["note"]["statements"]
    sid = visit["note"]["statements"][0]["id"]
    edited = client.put(
        f"/api/visits/pc-01/statements/{sid}", json={"text": "Home blood pressure readings around 145 to 150 systolic"}
    ).json()
    assert next(s for s in edited["note"]["statements"] if s["id"] == sid)["edited"]
    added = client.post(
        "/api/visits/pc-01/statements", json={"section": "plan", "text": "Discussed home blood pressure log"}
    ).json()
    assert any(s["origin"] == "clinician" for s in added["note"]["statements"])
    # Sign-off is refused until every section is accepted and every issue resolved
    assert client.post("/api/visits/pc-01/sign", json={"attestation": True}).status_code == 409
    visit = _resolve_everything(client, "pc-01", added)
    for section in ("subjective", "objective", "assessment", "plan"):
        r = client.post(f"/api/visits/pc-01/sections/{section}/accept")
        assert r.status_code == 200, r.text
        visit = r.json()
    code = visit["note"]["icd10"][0]["code"]
    assert client.post("/api/visits/pc-01/sign", json={"attestation": True}).status_code == 409  # no code accepted yet
    client.post("/api/visits/pc-01/codes", json={"code": code, "accepted": True})
    assert client.post("/api/visits/pc-01/sign", json={"attestation": False}).status_code == 422
    signed = client.post("/api/visits/pc-01/sign", json={"attestation": True})
    assert signed.status_code == 200 and signed.json()["status"] == "signed"
    assert client.post("/api/visits/pc-01/draft").status_code == 409  # locked
    assert client.put(f"/api/visits/pc-01/statements/{sid}", json={"text": "x"}).status_code == 409
    exported = client.post("/api/visits/pc-01/export/fhir")
    assert exported.status_code == 200, exported.text
    assert any(loc.startswith("Composition/") for loc in exported.json()["resources"])
    composition = client.get(
        "/mock-fhir/Composition/"
        + next(x for x in exported.json()["resources"] if x.startswith("Composition/")).split("/")[1]
    )
    assert composition.status_code == 200 and composition.json()["status"] == "final"
    pdf = client.get("/api/visits/pc-01/note.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    assert client.get("/api/visits/pc-01").json()["status"] == "exported"


def test_accept_section_with_open_errors_is_refused(client: TestClient) -> None:
    visit = _draft(client, "cd-03")
    sections_with_errors = {
        s["section"] for s in visit["note"]["statements"] if any(f["severity"] == "error" for f in s["flags"])
    }
    for section in sections_with_errors:
        assert client.post(f"/api/visits/cd-03/sections/{section}/accept").status_code == 409


def test_export_requires_signature(client: TestClient) -> None:
    _draft(client, "pc-02")
    assert client.post("/api/visits/pc-02/export/fhir").status_code == 409


def test_codes_search_and_add(client: TestClient) -> None:
    _draft(client, "pc-03")
    results = client.get("/api/codes/search", params={"q": "acute maxillary sinusitis"}).json()
    assert results and results[0]["code"].startswith("J01")
    assert client.post("/api/visits/pc-03/codes/add", json={"code": "J01.00"}).status_code in {200, 409}
    assert client.post("/api/visits/pc-03/codes/add", json={"code": "J01"}).status_code == 422


def test_audit_log_never_contains_identifiers(client: TestClient) -> None:
    visit = _draft(client, "pc-01")
    client.get("/api/visits/pc-01/masked")
    log = client.get("/api/audit").json()
    actions = {row["action"] for row in log}
    assert {"demo.seed", "note.draft", "visit.view_masked"} <= actions
    dump = json.dumps(log)
    assert visit["patient"]["name"] not in dump and visit["patient"]["dob"] not in dump


def test_create_visit_and_upload_audio(client: TestClient) -> None:
    created = client.post(
        "/api/visits", json={"title": "Cough", "patient": {"name": "Rowan Ellery", "dob": "1990-01-02"}}
    ).json()
    vid = created["id"]
    r = client.post(f"/api/visits/{vid}/audio", files={"file": ("a.webm", b"\x1aE\xdf\xa3fake", "audio/webm")})
    assert r.status_code == 200 and r.json()["has_audio"]
    audio = client.get(f"/api/visits/{vid}/audio")
    assert audio.content.startswith(b"\x1aE")
    # Speech recognition is off in the default (offline) configuration: a clear 409, not a crash
    assert client.post(f"/api/visits/{vid}/transcribe").status_code == 409


def test_auth_tokens_and_roles(settings: Settings) -> None:
    s = settings.model_copy(update={"auth_tokens": ["clinician:Dr. Ada:tok-c", "viewer:Sam:tok-v"]})
    svc = build_services(s, llm=ResilientModel(FakeScribe()))
    client = TestClient(create_app(s, svc))
    assert client.get("/api/visits").status_code == 401
    assert client.get("/api/visits", headers={"Authorization": "Bearer tok-v"}).status_code == 200
    assert client.post("/api/visits/pc-01/draft", headers={"Authorization": "Bearer tok-v"}).status_code == 403
    assert client.post("/api/visits/pc-01/draft", headers={"Authorization": "Bearer tok-c"}).status_code == 200
    assert svc.store.audit_log(limit=1)[0]["actor"] == "Dr. Ada"


def test_data_is_encrypted_at_rest(settings: Settings, services: Services) -> None:
    raw = settings.db_path.read_bytes()
    visit = services.visit("pc-01")
    assert visit.patient.name.encode() not in raw
    assert visit.transcript is not None and visit.transcript.turns[3].text[:30].encode() not in raw
