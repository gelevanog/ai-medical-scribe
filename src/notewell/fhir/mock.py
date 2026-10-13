"""A minimal in-memory FHIR R4 server for the demo (POST a transaction Bundle, GET resources back).

It implements just enough of the RESTful API for the export flow: transaction bundles with urn:uuid references
rewritten to server ids, `ifNoneExist` for conditional creates, read by id and a simple search by subject/patient.
For a real server, run HAPI FHIR (`docker compose --profile hapi up`) and point NOTEWELL_FHIR_BASE_URL at it.
"""

from __future__ import annotations

import copy
import itertools
import json
import threading
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

FHIR_JSON = "application/fhir+json"


class MockFHIRServer:
    def __init__(self) -> None:
        self.resources: dict[str, dict[str, dict[str, Any]]] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def _store(self, resource: dict[str, Any]) -> dict[str, Any]:
        rtype = resource["resourceType"]
        rid = str(next(self._ids))
        stored = {**resource, "id": rid, "meta": {**resource.get("meta", {}), "versionId": "1",
                  "lastUpdated": datetime.now(UTC).isoformat(timespec="seconds")}}  # fmt: skip
        self.resources.setdefault(rtype, {})[rid] = stored
        return stored

    def _find(self, rtype: str, criteria: str) -> dict[str, Any] | None:
        key, _, value = criteria.partition("=")
        if key != "identifier":
            return None
        system, _, code = value.partition("|")
        for res in self.resources.get(rtype, {}).values():
            for ident in res.get("identifier", []):
                if ident.get("value") == code and (not system or ident.get("system") == system):
                    return res
        return None

    def transaction(self, bundle: dict[str, Any]) -> dict[str, Any]:
        if bundle.get("resourceType") != "Bundle" or bundle.get("type") not in {"transaction", "batch"}:
            raise ValueError("expected a transaction Bundle")
        with self._lock:
            mapping: dict[str, str] = {}
            responses: list[dict[str, Any]] = []
            created: list[tuple[str, dict[str, Any]]] = []
            for entry in bundle.get("entry", []):
                resource = copy.deepcopy(entry["resource"])
                request = entry.get("request", {})
                existing = self._find(resource["resourceType"], request["ifNoneExist"]) if request.get("ifNoneExist") else None
                if existing is not None:
                    mapping[entry.get("fullUrl", "")] = f"{existing['resourceType']}/{existing['id']}"
                    responses.append({"response": {"status": "200 OK", "location": mapping[entry.get('fullUrl', '')]}})
                    continue
                stored = self._store(resource)
                mapping[entry.get("fullUrl", "")] = f"{stored['resourceType']}/{stored['id']}"
                created.append((stored["resourceType"], stored))
                responses.append({"response": {"status": "201 Created", "location": f"{stored['resourceType']}/{stored['id']}/_history/1"}})
            for _, stored in created:
                text = json.dumps(stored)
                for urn, ref in mapping.items():
                    if urn:
                        text = text.replace(f'"{urn}"', f'"{ref}"')
                stored.clear()
                stored.update(json.loads(text))
        return {"resourceType": "Bundle", "type": "transaction-response", "entry": responses}

    def read(self, rtype: str, rid: str) -> dict[str, Any] | None:
        return self.resources.get(rtype, {}).get(rid)

    def search(self, rtype: str, params: dict[str, str]) -> dict[str, Any]:
        items = list(self.resources.get(rtype, {}).values())
        for key in ("subject", "patient"):
            if key in params:
                ref = params[key] if "/" in params[key] else f"Patient/{params[key]}"
                items = [r for r in items if (r.get("subject") or r.get("patient") or {}).get("reference") == ref]
        return {"resourceType": "Bundle", "type": "searchset", "total": len(items), "entry": [{"resource": r} for r in items]}


def router(server: MockFHIRServer) -> APIRouter:
    api = APIRouter(prefix="/mock-fhir", tags=["mock FHIR server"])

    @api.get("/metadata")
    def metadata() -> JSONResponse:
        return JSONResponse(
            {"resourceType": "CapabilityStatement", "status": "active", "fhirVersion": "4.0.1", "kind": "instance",
             "format": ["json"], "software": {"name": "Notewell mock FHIR server"}},  # fmt: skip
            media_type=FHIR_JSON,
        )

    @api.post("")
    @api.post("/")
    async def transaction(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            return JSONResponse(server.transaction(body), media_type=FHIR_JSON)
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, f"invalid bundle: {exc}") from exc

    @api.get("/{rtype}/{rid}")
    def read(rtype: str, rid: str) -> JSONResponse:
        found = server.read(rtype, rid)
        if found is None:
            raise HTTPException(404, f"{rtype}/{rid} not found")
        return JSONResponse(found, media_type=FHIR_JSON)

    @api.get("/{rtype}")
    def search(rtype: str, request: Request) -> JSONResponse:
        return JSONResponse(server.search(rtype, dict(request.query_params)), media_type=FHIR_JSON)

    return api
