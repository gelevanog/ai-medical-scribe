"""Export signed notes to a real HAPI FHIR R4 server and validate every resource with $validate.

    docker compose --profile hapi up -d hapi        # or: docker run -p 8090:8080 hapiproject/hapi:v8.12.0-1
    uv run python scripts/hapi_check.py http://127.0.0.1:8090/fhir pd-01 cd-03 mh-02

Writes results/fhir/hapi_check.json: the created resource locations and the OperationOutcome issues per resource.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import httpx

from notewell.config import Settings, get_settings
from notewell.fhir.export import validate_bundle
from notewell.models import SECTIONS
from notewell.services import Services, build_services


def sign(svc: Services, vid: str) -> None:
    visit = svc.visit(vid)
    note = visit.note
    assert note is not None, f"{vid} has no note (seed the demo notes first)"
    for s in note.statements:
        for f in s.flags:
            if f.severity == "error":
                svc.acknowledge(vid, s.id, f.kind, "hapi-check")
    for i, m in enumerate(note.medications):
        for f in m.flags:
            if f.severity == "error":
                svc.acknowledge(vid, f"med:{i}", f.kind, "hapi-check")
    for c in [*note.icd10, *note.procedures, *([note.em] if note.em else [])]:
        for f in c.flags:
            if f.severity == "error":
                svc.acknowledge(vid, f"code:{c.code}", f.kind, "hapi-check")
    for f in note.avs_flags:
        if f.severity == "error":
            svc.acknowledge(vid, "avs", f.kind, "hapi-check")
    for c in svc.visit(vid).note.icd10[:2]:  # type: ignore[union-attr]
        if c.valid:
            svc.set_code(vid, "ICD-10-CM", c.code, True, "hapi-check")
    for section in SECTIONS:
        svc.accept_section(vid, section, "hapi-check")
    svc.sign(vid, "Dr. Hapi Check", attestation=True)


def main() -> int:
    base = sys.argv[1].rstrip("/")
    ids = sys.argv[2:] or ["pd-01"]
    tmp = Path(tempfile.mkdtemp())
    settings: Settings = get_settings().model_copy(update={"data_dir": tmp, "fhir_base_url": base, "llm_ledger": None})
    svc = build_services(settings)
    report: dict[str, object] = {"server": base, "visits": {}}
    with httpx.Client(timeout=60) as client:
        meta = client.get(f"{base}/metadata").json()
        report["software"] = meta.get("software")
        report["fhirVersion"] = meta.get("fhirVersion")
        for vid in ids:
            sign(svc, vid)
            bundle = svc.bundle(vid)
            local_errors = validate_bundle(bundle)
            result = svc.export_fhir(vid, "hapi-check")
            issues: dict[str, list[str]] = {}
            for location in result["resources"]:
                path = location.split("/_history")[0]
                resource = client.get(f"{base}/{path}").json()
                outcome = client.post(f"{base}/{path.split('/')[0]}/$validate", json=resource,
                                      headers={"Content-Type": "application/fhir+json"}).json()  # fmt: skip
                problems = [
                    f"{i.get('severity')}: {i.get('diagnostics', '')[:200]}"
                    for i in outcome.get("issue", [])
                    if i.get("severity") in {"error", "fatal"}
                ]
                issues[path] = problems
            report["visits"][vid] = {  # type: ignore[index]
                "resources": result["resources"],
                "local_validation_errors": local_errors,
                "hapi_validate_errors": {k: v for k, v in issues.items() if v},
                "resources_without_errors": sum(1 for v in issues.values() if not v),
            }
            print(
                vid,
                len(result["resources"]),
                "resources;",
                sum(1 for v in issues.values() if v),
                "with $validate errors",
            )
    out = Path("results/fhir/hapi_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
