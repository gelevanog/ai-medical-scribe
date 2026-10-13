"""The application layer shared by the HTTP API, the CLI and the MCP server: visits, transcription, drafting,
review, sign-off and export, with every state change written to the audit log."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from notewell.codes.icd10 import CodeSet, dotted, load_codeset
from notewell.config import Settings
from notewell.crypto import Cipher
from notewell.dataset import Case, load_cases
from notewell.fhir.export import build_bundle, validate_bundle
from notewell.fhir.mock import MockFHIRServer
from notewell.llm.base import ChatModel
from notewell.llm.factory import build_chat_model, resilient
from notewell.logging_config import get_logger
from notewell.models import SECTIONS, CodeSuggestion, Evidence, Flag, Note, Statement, Transcript, Visit
from notewell.pdf import render_pdf
from notewell.phi.masker import MaskedTranscript, Masker, registered_from
from notewell.scribe.pipeline import Scribe, ScribeError, regrade
from notewell.store import Store

log = get_logger(__name__)


class ServiceError(RuntimeError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Services:
    settings: Settings
    store: Store
    codeset: CodeSet
    llm: ChatModel
    masker: Masker = field(default_factory=Masker)
    fhir_mock: MockFHIRServer = field(default_factory=MockFHIRServer)
    cases: dict[str, Case] = field(default_factory=dict)
    _asr: Any = None

    # ---- construction --------------------------------------------------------------------------------------------
    @property
    def scribe(self) -> Scribe:
        return Scribe(
            self.llm,
            self.codeset,
            masker=self.masker,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            code_candidates=self.settings.code_candidates,
            block_on_phi=self.settings.block_on_phi,
        )

    # ---- visits --------------------------------------------------------------------------------------------------
    def seed_demo(self, *, notes_dir: Path | None = None) -> int:
        """Load the synthetic visits; attach pre-computed notes from an evaluation run when available."""
        existing = self.store.visit_ids()
        added = 0
        for case in self.cases.values():
            if case.id in existing:
                continue
            audio = f"{case.id}.wav" if (self.settings.audio_dir / f"{case.id}.wav").exists() else None
            visit = case.visit(audio=audio)
            visit.transcript = case.transcript()
            visit.status = "transcribed"
            visit.created_at = datetime.now(UTC)
            meta = self.settings.audio_dir / f"{case.id}.json"
            if meta.exists():
                timings = {t["idx"]: t for t in json.loads(meta.read_text(encoding="utf-8"))["turns"]}
                for turn in visit.transcript.turns:
                    if turn.idx in timings:
                        turn.start, turn.end = timings[turn.idx]["start"], timings[turn.idx]["end"]
            if notes_dir is not None and (notes_dir / f"{case.id}.json").exists():
                saved = json.loads((notes_dir / f"{case.id}.json").read_text(encoding="utf-8"))
                visit.note = Note.model_validate(saved["note"])
                visit.status = "drafted"
                if saved.get("vault"):
                    from notewell.phi.masker import Vault

                    self.store.save_vault(case.id, Vault.from_dict(saved["vault"]))
            self.store.save_visit(visit)
            added += 1
        if added:
            self.store.audit("system", "demo.seed", None, visits=added)
        return added

    def visits(self) -> list[Visit]:
        return self.store.list_visits()

    def visit(self, visit_id: str) -> Visit:
        visit = self.store.get_visit(visit_id)
        if visit is None:
            raise ServiceError(f"visit {visit_id!r} not found", 404)
        return visit

    def create_visit(self, data: dict[str, Any], actor: str) -> Visit:
        visit = Visit.model_validate(
            {
                "id": data.get("id") or f"v-{uuid.uuid4().hex[:8]}",
                "title": data.get("title") or "New visit",
                "specialty": data.get("specialty") or "primary_care",
                "visit_type": data.get("visit_type") or "established",
                "visit_date": data.get("visit_date") or datetime.now(UTC).date().isoformat(),
                "patient": data["patient"],
                "clinician": data.get("clinician") or {"name": self.settings.demo_user.split(" (")[0]},
                "synthetic": bool(data.get("synthetic", True)),
                "created_at": datetime.now(UTC),
            }
        )
        self.store.save_visit(visit)
        self.store.audit(actor, "visit.create", visit.id)
        return visit

    def attach_audio(self, visit_id: str, data: bytes, mime: str, actor: str) -> Visit:
        visit = self.visit(visit_id)
        self.store.save_upload(visit_id, data, mime)
        visit.audio = f"upload:{visit_id}"
        self.store.save_visit(visit)
        self.store.audit(actor, "audio.upload", visit_id, bytes=len(data), mime=mime)
        return visit

    def audio(self, visit_id: str) -> tuple[bytes, str]:
        visit = self.visit(visit_id)
        if visit.audio and visit.audio.startswith("upload:"):
            found = self.store.get_upload(visit_id)
            if found:
                return found
        if visit.audio:
            path = self.settings.audio_dir / visit.audio
            if path.exists():
                return path.read_bytes(), "audio/wav"
        raise ServiceError("no audio for this visit", 404)

    # ---- transcription -------------------------------------------------------------------------------------------
    def transcribe(self, visit_id: str, actor: str, *, force_asr: bool = False) -> Visit:
        visit = self.visit(visit_id)
        use_asr = self.settings.stt_provider == "whisper" or force_asr
        if not use_asr:
            case = self.cases.get(visit_id)
            if case is None:
                raise ServiceError(
                    "speech recognition is off (NOTEWELL_STT_PROVIDER=fake); install the speech extra and set "
                    "NOTEWELL_STT_PROVIDER=whisper to transcribe recorded audio",
                    409,
                )
            visit.transcript = case.transcript()
        else:
            visit.transcript = self._run_asr(visit)
        visit.status = "transcribed"
        visit.note = None
        self.store.save_visit(visit)
        self.store.audit(
            actor, "visit.transcribe", visit_id, source=visit.transcript.source, turns=len(visit.transcript.turns)
        )
        return visit

    def _run_asr(self, visit: Visit) -> Transcript:
        import tempfile

        from notewell.speech.asr import WhisperASR, decode_audio

        if self._asr is None:
            self._asr = WhisperASR(
                self.settings.stt_model,
                threads=self.settings.stt_threads,
                medical_prompt=self.settings.stt_medical_prompt,
            )
        data, mime = self.audio(visit.id)
        suffix = {"audio/wav": ".wav", "audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mpeg": ".mp3"}.get(
            mime.split(";")[0], ".bin"
        )
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            audio = decode_audio(Path(tmp.name))
        second = "caregiver" if visit.specialty == "pediatrics" else "patient"
        transcript: Transcript = self._asr.transcribe(audio, second_speaker=second)
        return transcript

    def masked(self, visit_id: str) -> MaskedTranscript:
        visit = self.visit(visit_id)
        vault = self.store.get_vault(visit_id)
        try:
            return self.scribe.mask(visit, vault)
        except ScribeError as exc:
            raise ServiceError(str(exc), 409) from exc

    # ---- drafting ------------------------------------------------------------------------------------------------
    def draft(self, visit_id: str, actor: str) -> Visit:
        visit = self.visit(visit_id)
        if visit.status in {"signed", "exported"}:
            raise ServiceError("the note is signed; it cannot be re-drafted", 409)
        try:
            result = self.scribe.draft(visit)
        except ScribeError as exc:
            raise ServiceError(str(exc), 409) from exc
        visit.note = result.note
        visit.status = "drafted"
        self.store.save_vault(visit_id, result.masked.vault)
        self.store.save_visit(visit)
        flags = [f for f in _all_flags(result.note) if f.severity == "error"]
        self.store.audit(
            actor,
            "note.draft",
            visit_id,
            model=result.note.model,
            statements=len(result.note.statements),
            error_flags=len(flags),
            phi_masked=result.note.phi,
            usage=result.note.usage,
        )
        return visit

    # ---- review --------------------------------------------------------------------------------------------------
    def _editable(self, visit_id: str) -> tuple[Visit, Note]:
        visit = self.visit(visit_id)
        if visit.note is None:
            raise ServiceError("no draft note yet", 409)
        if visit.status in {"signed", "exported"}:
            raise ServiceError("the note is signed and locked", 409)
        return visit, visit.note

    def _save_note(self, visit: Visit, note: Note, actor: str, action: str, **detail: Any) -> Visit:
        visit.note = note
        self.store.save_visit(visit)
        self.store.audit(actor, action, visit.id, **detail)
        return visit

    def edit_statement(self, visit_id: str, statement_id: str, text: str, actor: str) -> Visit:
        visit, note = self._editable(visit_id)
        stmt = _find_statement(note, statement_id)
        stmt.text = text.strip()
        stmt.edited = True
        note.section_status[stmt.section] = "draft"
        note = regrade(note, visit, self.codeset)
        return self._save_note(visit, note, actor, "note.edit", statement=statement_id)

    def add_statement(self, visit_id: str, section: str, text: str, actor: str) -> Visit:
        visit, note = self._editable(visit_id)
        if section not in SECTIONS:
            raise ServiceError(f"unknown section {section!r}")
        sid = f"{section[0]}c{sum(1 for s in note.statements if s.section == section) + 1}"
        note.statements.append(Statement(id=sid, section=section, text=text.strip(), origin="clinician", edited=True))
        note.section_status[section] = "draft"
        note = regrade(note, visit, self.codeset)
        return self._save_note(visit, note, actor, "note.add", statement=sid)

    def delete_statement(self, visit_id: str, statement_id: str, actor: str) -> Visit:
        visit, note = self._editable(visit_id)
        stmt = _find_statement(note, statement_id)
        note.statements = [s for s in note.statements if s.id != statement_id]
        note.section_status[stmt.section] = "draft"
        return self._save_note(visit, note, actor, "note.delete", statement=statement_id)

    def acknowledge(self, visit_id: str, statement_id: str, kind: str, actor: str) -> Visit:
        visit, note = self._editable(visit_id)
        found = False
        for flag in _flags_of(note, statement_id):
            if flag.kind == kind:
                flag.acknowledged = True
                found = True
        if not found:
            raise ServiceError(f"no {kind!r} flag on {statement_id!r}", 404)
        return self._save_note(visit, note, actor, "flag.acknowledge", target=statement_id, kind=kind)

    def accept_section(self, visit_id: str, section: str, actor: str, *, accepted: bool = True) -> Visit:
        visit, note = self._editable(visit_id)
        if section not in SECTIONS:
            raise ServiceError(f"unknown section {section!r}")
        if accepted:
            open_errors = [
                f for s in note.section(section) for f in s.flags if f.severity == "error" and not f.acknowledged
            ]
            if open_errors:
                raise ServiceError(
                    f"{len(open_errors)} open issue(s) in {section}: fix or acknowledge them before accepting", 409
                )
        note.section_status[section] = "accepted" if accepted else "draft"
        return self._save_note(visit, note, actor, "section.accept" if accepted else "section.reopen", section=section)

    def set_code(self, visit_id: str, system: str, code: str, accepted: bool | None, actor: str) -> Visit:
        visit, note = self._editable(visit_id)
        pool: list[CodeSuggestion] = [*note.icd10, *note.procedures, *([note.em] if note.em else [])]
        matches = [c for c in pool if c.code == code and (system == "any" or c.system == system)]
        if not matches:
            raise ServiceError(f"code {code} is not among the suggestions", 404)
        for c in matches:
            if accepted and any(f.severity == "error" and not f.acknowledged for f in c.flags):
                raise ServiceError(f"{code} has an open error ({c.flags[0].message}); acknowledge it first", 409)
            c.accepted = accepted
        return self._save_note(visit, note, actor, "code.review", code=code, accepted=accepted)

    def add_code(self, visit_id: str, code: str, actor: str) -> Visit:
        visit, note = self._editable(visit_id)
        validation = self.codeset.validate(code)
        if validation.status != "valid":
            raise ServiceError(validation.message or f"{code} is not a valid billable ICD-10-CM code", 422)
        if any(c.code == validation.code for c in note.icd10):
            raise ServiceError(f"{validation.code} is already on the note", 409)
        note.icd10.append(
            CodeSuggestion(
                system="ICD-10-CM",
                code=validation.code,
                description=validation.title,
                rank=len(note.icd10) + 1,
                rationale="Added by the clinician",
                valid=True,
                billable=True,
                accepted=True,
            )
        )
        return self._save_note(visit, note, actor, "code.add", code=validation.code)

    # ---- sign-off and export -------------------------------------------------------------------------------------
    def sign(self, visit_id: str, actor: str, *, attestation: bool) -> Visit:
        visit, note = self._editable(visit_id)
        if not attestation:
            raise ServiceError("sign-off requires the clinician's attestation that they reviewed the note", 422)
        pending = [s for s in SECTIONS if note.section_status.get(s) != "accepted"]
        if pending:
            raise ServiceError(f"accept every section first (pending: {', '.join(pending)})", 409)
        open_errors = note.open_flags()
        if open_errors:
            raise ServiceError(f"{len(open_errors)} open issue(s) remain; fix or acknowledge them", 409)
        if not any(c.accepted for c in note.icd10):
            raise ServiceError("accept at least one diagnosis code (or add one)", 409)
        note.signed_by = actor
        note.signed_at = datetime.now(UTC)
        visit.status = "signed"
        return self._save_note(visit, note, actor, "note.sign")

    def bundle(self, visit_id: str) -> dict[str, Any]:
        visit = self.visit(visit_id)
        if visit.note is None or visit.note.signed_at is None:
            raise ServiceError("only a signed note can be exported", 409)
        return build_bundle(
            visit, visit.note, pdf=render_pdf(visit, visit.note), organization=self.settings.organization_name
        )

    def export_fhir(self, visit_id: str, actor: str) -> dict[str, Any]:
        visit = self.visit(visit_id)
        bundle = self.bundle(visit_id)
        problems = validate_bundle(bundle)
        if problems:
            raise ServiceError("FHIR validation failed: " + "; ".join(problems[:3]), 422)
        base = self.settings.fhir_base_url.rstrip("/")
        if base.endswith("/mock-fhir") and base.startswith(("http://127.0.0.1", "http://localhost")):
            response = self.fhir_mock.transaction(bundle)
        else:
            try:
                r = httpx.post(base, json=bundle, headers={"Content-Type": "application/fhir+json"}, timeout=30)
            except httpx.HTTPError as exc:
                raise ServiceError(f"FHIR server not reachable at {base}: {exc}", 502) from exc
            if r.status_code >= 400:
                raise ServiceError(f"FHIR server answered {r.status_code}: {r.text[:300]}", 502)
            response = r.json()
        locations = [e.get("response", {}).get("location", "") for e in response.get("entry", [])]
        visit.status = "exported"
        self.store.save_visit(visit)
        self.store.audit(actor, "note.export_fhir", visit_id, server=_redact_url(base), resources=len(locations))
        return {"server": base, "resources": locations, "bundle_entries": len(bundle["entry"])}

    def pdf(self, visit_id: str, actor: str) -> bytes:
        visit = self.visit(visit_id)
        if visit.note is None:
            raise ServiceError("no note yet", 409)
        self.store.audit(actor, "note.pdf", visit_id)
        return render_pdf(visit, visit.note)


def _redact_url(url: str) -> str:
    return re.sub(r"//[^@/]+@", "//", url)


def _find_statement(note: Note, statement_id: str) -> Statement:
    for s in note.statements:
        if s.id == statement_id:
            return s
    raise ServiceError(f"statement {statement_id!r} not found", 404)


def _flags_of(note: Note, target: str) -> list[Flag]:
    for s in note.statements:
        if s.id == target:
            return s.flags
    for i, m in enumerate(note.medications):
        if target in {f"med:{i}", f"med:{m.name}"}:
            return m.flags
    for c in [*note.icd10, *note.procedures, *([note.em] if note.em else [])]:
        if target in {f"code:{c.code}", c.code}:
            return c.flags
    if target == "avs":
        return note.avs_flags
    raise ServiceError(f"{target!r} not found", 404)


def _all_flags(note: Note) -> list[Flag]:
    flags = [f for s in note.statements for f in s.flags] + [f for m in note.medications for f in m.flags]
    flags += [f for c in [*note.icd10, *note.procedures, *([note.em] if note.em else [])] for f in c.flags]
    return flags + note.avs_flags


def evidence_payload(evidence: list[Evidence]) -> list[dict[str, Any]]:
    return [e.model_dump() for e in evidence]


def build_services(settings: Settings, *, llm: ChatModel | None = None, seed: bool | None = None) -> Services:
    cipher = Cipher.from_settings(settings.data_key, settings.data_dir)
    store = Store(settings.db_path, cipher)
    codeset = load_codeset(str(settings.codes_dir))
    model = llm or resilient(build_chat_model(settings), settings, tag="app")
    cases = {c.id: c for c in load_cases(settings.dataset_dir)} if settings.dataset_dir.exists() else {}
    services = Services(settings=settings, store=store, codeset=codeset, llm=model, cases=cases)
    if settings.seed_demo if seed is None else seed:
        notes = settings.results_dir / "runs" / settings.demo_notes_run / "notes"
        services.seed_demo(notes_dir=notes if notes.exists() else None)
    return services


SignMode = Literal["sign"]
__all__ = ["ServiceError", "Services", "build_services", "dotted", "registered_from"]
