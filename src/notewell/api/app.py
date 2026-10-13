"""HTTP API for the review UI (Next.js) and integrations. Every request acts as an authenticated user (demo mode: the
demo clinician), and every state change is written to the audit log by the service layer."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from notewell.config import Settings, get_settings
from notewell.dataset import SPECIALTY_LABELS
from notewell.fhir import mock
from notewell.logging_config import configure_logging, get_logger
from notewell.models import Patient, Visit
from notewell.services import ServiceError, Services, build_services

log = get_logger(__name__)
MAX_UPLOAD_BYTES = 60 * 1024 * 1024


@dataclass(frozen=True)
class User:
    name: str
    role: str


class EditBody(BaseModel):
    text: str


class AddBody(BaseModel):
    section: str
    text: str


class CodeBody(BaseModel):
    system: str = "any"
    code: str
    accepted: bool | None = True


class AddCodeBody(BaseModel):
    code: str


class AckBody(BaseModel):
    target: str
    kind: str


class SignBody(BaseModel):
    attestation: bool = False


class CreateVisitBody(BaseModel):
    title: str = "New visit"
    specialty: str = "primary_care"
    visit_type: str = "established"
    visit_date: str | None = None
    patient: Patient


def visit_summary(v: Visit) -> dict[str, Any]:
    note = v.note
    flags = 0
    if note is not None:
        flags = len(note.open_flags())
    primary = next((c for c in (note.icd10 if note else [])), None)
    return {
        "id": v.id,
        "title": v.title,
        "specialty": v.specialty,
        "specialty_label": SPECIALTY_LABELS.get(v.specialty, v.specialty),
        "visit_type": v.visit_type,
        "visit_date": v.visit_date,
        "status": v.status,
        "patient": {"name": v.patient.name, "dob": v.patient.dob, "sex": v.patient.sex},
        "clinician": v.clinician.name,
        "has_audio": bool(v.audio),
        "duration_s": (v.transcript.turns[-1].end if v.transcript and v.transcript.turns and v.transcript.turns[-1].end else None),
        "open_issues": flags,
        "primary_code": {"code": primary.code, "description": primary.description} if primary else None,
        "model": note.model if note else None,
        "synthetic": v.synthetic,
    }


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)
    app = FastAPI(
        title="Notewell API",
        description="AI medical scribe demo: synthetic data only, not a medical device, not for clinical use.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )
    state: dict[str, Services] = {}

    def svc() -> Services:
        if "services" not in state:
            state["services"] = services or build_services(settings)
        return state["services"]

    tokens: dict[str, User] = {}
    for triple in settings.auth_tokens:
        role, name, token = triple.split(":", 2)
        tokens[token] = User(name=name, role=role)

    def current_user(authorization: Annotated[str | None, Header()] = None) -> User:
        if not tokens:
            return User(name=settings.demo_user, role="clinician")
        token = (authorization or "").removeprefix("Bearer ").strip()
        user = tokens.get(token)
        if user is None:
            raise HTTPException(401, "missing or invalid bearer token")
        return user

    def clinician(user: Annotated[User, Depends(current_user)]) -> User:
        if user.role != "clinician":
            raise HTTPException(403, "only a clinician may change or sign a note")
        return user

    CurrentUser = Annotated[User, Depends(current_user)]
    Clinician = Annotated[User, Depends(clinician)]

    @app.exception_handler(ServiceError)
    async def service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    app.include_router(mock.router(svc().fhir_mock if services else _LazyMock(svc)))  # type: ignore[arg-type]

    @app.get("/health")
    def health() -> dict[str, Any]:
        s = svc()
        return {
            "status": "ok",
            "llm": s.llm.label,
            "stt": settings.stt_provider,
            "codes": {"source": "full" if s.codeset.complete else "subset", "count": len(s.codeset.codes)},
            "fhir": settings.fhir_base_url,
            "demo_mode": not tokens,
        }

    @app.get("/api/me")
    def me(user: CurrentUser) -> dict[str, str]:
        return {"name": user.name, "role": user.role}

    @app.get("/api/visits")
    def list_visits(_: CurrentUser) -> list[dict[str, Any]]:
        return [visit_summary(v) for v in svc().visits()]

    @app.post("/api/visits")
    def create_visit(body: CreateVisitBody, user: Clinician) -> dict[str, Any]:
        visit = svc().create_visit(body.model_dump(), user.name)
        return visit_summary(visit)

    @app.get("/api/visits/{visit_id}")
    def get_visit(visit_id: str, _: CurrentUser) -> dict[str, Any]:
        visit = svc().visit(visit_id)
        return {**visit.model_dump(mode="json"), "summary": visit_summary(visit)}

    @app.get("/api/visits/{visit_id}/masked")
    def masked(visit_id: str, user: CurrentUser) -> dict[str, Any]:
        """Exactly what a cloud model sees: the masked transcript. Counts by type, never the values."""
        m = svc().masked(visit_id)
        svc().store.audit(user.name, "visit.view_masked", visit_id)
        return {"turns": [t.model_dump() for t in m.turns], "counts": m.vault.counts()}

    @app.post("/api/visits/{visit_id}/audio")
    async def upload_audio(visit_id: str, user: Clinician, file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "audio file too large")
        visit = svc().attach_audio(visit_id, data, file.content_type or "application/octet-stream", user.name)
        return visit_summary(visit)

    @app.get("/api/visits/{visit_id}/audio")
    def audio(visit_id: str, _: CurrentUser) -> Response:
        data, mime = svc().audio(visit_id)
        return Response(content=data, media_type=mime, headers={"Cache-Control": "private, max-age=3600"})

    def run(fn: Callable[[], Visit]) -> dict[str, Any]:
        visit = fn()
        return {**visit.model_dump(mode="json"), "summary": visit_summary(visit)}

    @app.post("/api/visits/{visit_id}/transcribe")
    def transcribe(visit_id: str, user: Clinician) -> dict[str, Any]:
        return run(lambda: svc().transcribe(visit_id, user.name))

    @app.post("/api/visits/{visit_id}/draft")
    def draft(visit_id: str, user: Clinician) -> dict[str, Any]:
        return run(lambda: svc().draft(visit_id, user.name))

    @app.put("/api/visits/{visit_id}/statements/{statement_id}")
    def edit(visit_id: str, statement_id: str, body: EditBody, user: Clinician) -> dict[str, Any]:
        return run(lambda: svc().edit_statement(visit_id, statement_id, body.text, user.name))

    @app.delete("/api/visits/{visit_id}/statements/{statement_id}")
    def delete(visit_id: str, statement_id: str, user: Clinician) -> dict[str, Any]:
        return run(lambda: svc().delete_statement(visit_id, statement_id, user.name))

    @app.post("/api/visits/{visit_id}/statements")
    def add(visit_id: str, body: AddBody, user: Clinician) -> dict[str, Any]:
        return run(lambda: svc().add_statement(visit_id, body.section, body.text, user.name))

    @app.post("/api/visits/{visit_id}/acknowledge")
    def acknowledge(visit_id: str, body: AckBody, user: Clinician) -> dict[str, Any]:
        return run(lambda: svc().acknowledge(visit_id, body.target, body.kind, user.name))

    @app.post("/api/visits/{visit_id}/sections/{section}/accept")
    def accept(visit_id: str, section: str, user: Clinician) -> dict[str, Any]:
        return run(lambda: svc().accept_section(visit_id, section, user.name))

    @app.post("/api/visits/{visit_id}/sections/{section}/reopen")
    def reopen(visit_id: str, section: str, user: Clinician) -> dict[str, Any]:
        return run(lambda: svc().accept_section(visit_id, section, user.name, accepted=False))

    @app.post("/api/visits/{visit_id}/codes")
    def review_code(visit_id: str, body: CodeBody, user: Clinician) -> dict[str, Any]:
        return run(lambda: svc().set_code(visit_id, body.system, body.code, body.accepted, user.name))

    @app.post("/api/visits/{visit_id}/codes/add")
    def add_code(visit_id: str, body: AddCodeBody, user: Clinician) -> dict[str, Any]:
        return run(lambda: svc().add_code(visit_id, body.code, user.name))

    @app.get("/api/codes/search")
    def search_codes(q: str, _: CurrentUser, limit: int = 12) -> list[dict[str, Any]]:
        return [{"code": c.dotted, "title": c.title, "score": round(s, 2)} for c, s in svc().codeset.search(q, limit=limit)]

    @app.post("/api/visits/{visit_id}/sign")
    def sign(visit_id: str, body: SignBody, user: Clinician) -> dict[str, Any]:
        return run(lambda: svc().sign(visit_id, user.name, attestation=body.attestation))

    @app.get("/api/visits/{visit_id}/fhir")
    def fhir_preview(visit_id: str, _: CurrentUser) -> dict[str, Any]:
        return svc().bundle(visit_id)

    @app.post("/api/visits/{visit_id}/export/fhir")
    def export_fhir(visit_id: str, user: Clinician) -> dict[str, Any]:
        return svc().export_fhir(visit_id, user.name)

    @app.get("/api/visits/{visit_id}/note.pdf")
    def note_pdf(visit_id: str, user: CurrentUser) -> Response:
        data = svc().pdf(visit_id, user.name)
        return Response(
            content=data,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="note-{visit_id}.pdf"'},
        )

    @app.get("/api/audit")
    def audit(_: CurrentUser, visit_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        return svc().store.audit_log(visit_id=visit_id, limit=limit)

    @app.get("/api/eval")
    def evaluation(_: CurrentUser) -> dict[str, Any]:
        path = settings.results_dir / "summary.json"
        if not path.exists():
            return {"available": False}
        return {"available": True, **json.loads(path.read_text(encoding="utf-8"))}

    return app


class _LazyMock:
    """Defers building the services (and the mock server inside them) until the first FHIR request."""

    def __init__(self, svc: Callable[[], Services]) -> None:
        self._svc = svc

    def __getattr__(self, name: str) -> Any:
        return getattr(self._svc().fhir_mock, name)


@lru_cache
def create_default_app() -> FastAPI:
    return create_app()
