"""The scribe pipeline: mask -> egress check -> draft note -> restore locally -> ground -> suggest codes -> guards.

Nothing identifying leaves the process: the model sees the masked transcript only; the egress check scans the final
payload for every known identifier (fail closed); the model's answer is restored with the vault in-process; every
statement is verified against the original transcript.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from notewell.codes.em import estimate
from notewell.codes.icd10 import CodeSet
from notewell.guards.avs import check_avs
from notewell.guards.coding import check_em, check_icd, check_procedures
from notewell.guards.grounding import Grounder
from notewell.guards.medications import check_medication
from notewell.guards.negation import check_statement, index_transcript
from notewell.kb import cpt
from notewell.llm.base import ChatModel, Completion, LLMError, Message, PolicyViolationError
from notewell.llm.resilient import ResilientModel
from notewell.logging_config import get_logger
from notewell.models import Allergy, Evidence, Flag, Medication, Note, Statement, Turn, Visit
from notewell.phi.egress import EgressGuard
from notewell.phi.masker import PLACEHOLDER_RE, MaskedTranscript, Masker, Vault, registered_from
from notewell.scribe.parse import ParseError, extract_json, parse_codes, parse_note
from notewell.scribe.prompts import codes_messages, note_messages, repair_message

log = get_logger(__name__)

Observer = Callable[[str, Sequence[Message]], None]


class ScribeError(RuntimeError):
    pass


@dataclass
class DraftResult:
    note: Note
    masked: MaskedTranscript
    egress: dict[str, int] = field(default_factory=dict)
    """Residual identifiers the egress check found and masked before sending (counts by type)."""
    payloads: int = 0


class Scribe:
    def __init__(
        self,
        llm: ChatModel,
        codeset: CodeSet,
        *,
        masker: Masker | None = None,
        max_tokens: int = 6000,
        temperature: float = 0.0,
        code_candidates: int = 25,
        block_on_phi: bool = True,
        observer: Observer | None = None,
    ) -> None:
        self.llm = llm
        self.codeset = codeset
        self.masker = masker or Masker()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.code_candidates = code_candidates
        self.block_on_phi = block_on_phi
        self.observer = observer

    # ---- masking ------------------------------------------------------------------------------------------------
    def mask(self, visit: Visit, vault: Vault | None = None) -> MaskedTranscript:
        if visit.transcript is None:
            raise ScribeError("the visit has no transcript yet")
        registered = registered_from(visit.patient, visit.clinician.name)
        return self.masker.mask_turns(visit.transcript.turns, registered, vault)

    def _call(
        self, messages: list[Message], guard: EgressGuard, vault: Vault, tag: str, residue: dict[str, int]
    ) -> tuple[Completion, list[Message]]:
        messages = self._enforce_egress(messages, guard, vault, residue)
        model: ChatModel = self.llm
        if isinstance(model, ResilientModel):
            model = model.with_options(tag=tag, egress_check=guard.check, observer=self.observer)
        else:
            if self.block_on_phi:
                guard.check(messages)
            if self.observer is not None:
                self.observer(model.label, messages)
        return model.complete(messages, max_tokens=self.max_tokens, temperature=self.temperature), messages

    def _enforce_egress(
        self, messages: list[Message], guard: EgressGuard, vault: Vault, residue: dict[str, int]
    ) -> list[Message]:
        """Mask what the egress check still finds (the masker missed it), then check again; fail closed."""
        report = guard.scan(messages)
        if report.clean:
            return messages
        for kind, n in report.hits.items():
            residue[kind] = residue.get(kind, 0) + n
        extra = sorted(report.found, key=len, reverse=True)
        kind = next(iter(report.hits), "ID")
        cleaned: list[Message] = []
        for message in messages:
            content = message["content"]
            for value in extra:
                if not value.strip():
                    continue
                placeholder = vault.add(value, "NAME" if kind.startswith("NAME") else kind)
                content = re.sub(rf"(?<!\w){re.escape(value)}(?!\w)", placeholder, content, flags=re.IGNORECASE)
            cleaned.append({"role": message["role"], "content": content})
        if not guard.scan(cleaned).clean and self.block_on_phi:
            raise PolicyViolationError("egress check: identifiers remain after masking; the request was not sent")
        log.warning("egress.residue_masked", kinds=sorted(report.hits))
        return cleaned

    def _json_call(
        self,
        messages: list[Message],
        guard: EgressGuard,
        vault: Vault,
        tag: str,
        residue: dict[str, int],
        usage: dict[str, int],
    ) -> dict[str, Any]:
        completion, sent = self._call(messages, guard, vault, tag, residue)
        usage["input_tokens"] = usage.get("input_tokens", 0) + completion.input_tokens
        usage["output_tokens"] = usage.get("output_tokens", 0) + completion.output_tokens
        usage["calls"] = usage.get("calls", 0) + 1
        try:
            return extract_json(completion.text)
        except ParseError as exc:
            retry = [*sent, {"role": "assistant", "content": completion.text[:4000]}, repair_message(str(exc))]
            completion, _ = self._call(retry, guard, vault, f"{tag}:repair", residue)
            usage["input_tokens"] += completion.input_tokens
            usage["output_tokens"] += completion.output_tokens
            usage["calls"] += 1
            try:
                return extract_json(completion.text)
            except ParseError as exc2:
                raise ScribeError(f"the model's answer is not valid JSON after one repair: {exc2}") from exc2

    # ---- the whole draft ----------------------------------------------------------------------------------------
    def draft(self, visit: Visit, *, vault: Vault | None = None) -> DraftResult:
        if visit.transcript is None or not visit.transcript.turns:
            raise ScribeError("the visit has no transcript yet")
        timings: dict[str, float] = {}
        usage: dict[str, int] = {}
        residue: dict[str, int] = {}
        t0 = time.perf_counter()
        masked = self.mask(visit, vault)
        registered = registered_from(visit.patient, visit.clinician.name)
        guard = EgressGuard(registered, masked.vault)
        timings["phi_masking"] = round(time.perf_counter() - t0, 3)

        t1 = time.perf_counter()
        note_data = self._json_call(
            note_messages(transcript=masked.text(), specialty=visit.specialty, visit_type=visit.visit_type),
            guard,
            masked.vault,
            f"draft_note:{visit.id}",
            residue,
            usage,
        )
        timings["llm_note"] = round(time.perf_counter() - t1, 3)

        t2 = time.perf_counter()
        statements, meds, allergies, avs = parse_note(note_data)
        vault = masked.vault
        unknown = self._restore(statements, meds, allergies, vault)
        avs_restored = vault.restore(avs)
        original_turns = visit.transcript.turns
        statements, meds, allergies = self._verify(statements, meds, allergies, original_turns, visit)
        timings["guards_note"] = round(time.perf_counter() - t2, 3)

        t3 = time.perf_counter()
        note_text_masked = "\n".join(
            f"{s.section[0].upper()}: {self.masker.mask_text(s.text, registered, vault)}" for s in statements
        )
        candidates = self._candidates(statements)
        allowed = [(c, v["summary"]) for c, v in cpt().items() if v["group"] in {"procedures", "psychiatry"}]
        codes_data = self._json_call(
            codes_messages(
                note_text=note_text_masked,
                transcript=masked.text(),
                visit_type=visit.visit_type,
                candidates=candidates,
                allowed_cpt=allowed,
            ),
            guard,
            vault,
            f"suggest_codes:{visit.id}",
            residue,
            usage,
        )
        timings["llm_codes"] = round(time.perf_counter() - t3, 3)

        t4 = time.perf_counter()
        note = self._assemble_codes(visit, statements, meds, allergies, avs_restored, codes_data, vault, original_turns)
        if unknown:
            note.avs_flags.append(
                Flag(
                    kind="placeholder",
                    severity="warning",
                    message=f"The model used placeholders that do not exist ({', '.join(sorted(set(unknown)))}).",
                )
            )
        timings["guards_codes"] = round(time.perf_counter() - t4, 3)
        timings["total"] = round(time.perf_counter() - t0, 3)
        note.timings = timings
        note.usage = usage
        note.phi = {**vault.counts(), **{f"egress_residue_{k}": v for k, v in residue.items()}}
        note.model = getattr(self.llm, "last_served", None) or self.llm.label
        note.created_at = datetime.now(UTC)
        return DraftResult(note=note, masked=masked, egress=residue)

    # ---- steps --------------------------------------------------------------------------------------------------
    @staticmethod
    def _restore(
        statements: list[Statement], meds: list[Medication], allergies: list[Allergy], vault: Vault
    ) -> list[str]:
        unknown: list[str] = []

        def restore(text: str) -> str:
            unknown.extend(vault.unknown_placeholders(text))
            return vault.restore(text)

        for s in statements:
            s.text = restore(s.text)
            for e in s.evidence:
                e.quote = restore(e.quote)
        for m in meds:
            m.name, m.dose, m.frequency = restore(m.name), restore(m.dose), restore(m.frequency)
            for e in m.evidence:
                e.quote = restore(e.quote)
        for a in allergies:
            a.substance, a.reaction = restore(a.substance), restore(a.reaction)
            for e in a.evidence:
                e.quote = restore(e.quote)
        return unknown

    @staticmethod
    def _verify(
        statements: list[Statement],
        meds: list[Medication],
        allergies: list[Allergy],
        turns: list[Turn],
        visit: Visit,
    ) -> tuple[list[Statement], list[Medication], list[Allergy]]:
        grounder = Grounder(turns)
        index = index_transcript(turns)
        checked: list[Statement] = []
        for s in statements:
            g = grounder.ground(s)
            for issue in check_statement(g.text, index, section=g.section):
                label = {
                    "negation": "The transcript says the opposite",
                    "attribution": "This was said about someone else (family or another person)",
                    "hypothetical": "This was only mentioned as a possibility or a precaution",
                }[issue.kind]
                g.flags.append(
                    Flag(
                        kind=issue.kind,
                        severity="error",
                        message=f"{label}: {issue.surface!r} (turns {', '.join(map(str, issue.turns[:4]))}).",
                        detail={"concept": issue.concept, "turns": issue.turns},
                    )
                )
            checked.append(g)
        pediatric = _is_pediatric(visit)
        for m in meds:
            m.evidence = [e for e in (grounder.verify_quote(ev) for ev in m.evidence) if e is not None]
            m.flags = check_medication(m, turns=turns, allergies=allergies, pediatric=pediatric)
        for a in allergies:
            a.evidence = [e for e in (grounder.verify_quote(ev) for ev in a.evidence) if e is not None]
        return checked, meds, allergies

    def _candidates(self, statements: list[Statement]) -> list[tuple[str, str]]:
        seen: dict[str, str] = {}
        queries = [s.text for s in statements if s.section == "assessment"] or [
            s.text for s in statements if s.section == "subjective"
        ][:3]
        per_query = max(4, self.code_candidates // max(1, len(queries)))
        for q in queries:
            for code, _ in self.codeset.search(q, limit=per_query):
                seen.setdefault(code.dotted, code.title)
                if len(seen) >= self.code_candidates:
                    return list(seen.items())
        return list(seen.items())

    def _assemble_codes(
        self,
        visit: Visit,
        statements: list[Statement],
        meds: list[Medication],
        allergies: list[Allergy],
        avs: str,
        codes_data: dict[str, Any],
        vault: Vault,
        turns: list[Turn],
    ) -> Note:
        icd_raw, em_raw, proc_raw = parse_codes(codes_data)
        for item in [*icd_raw, *proc_raw, *([em_raw] if em_raw else [])]:
            for key in ("rationale", "description"):
                if isinstance(item.get(key), str):
                    item[key] = vault.restore(item[key])
            for ev in item.get("evidence") or []:
                if isinstance(ev, dict) and isinstance(ev.get("quote"), str):
                    ev["quote"] = vault.restore(ev["quote"])
        grounder = Grounder(turns)
        index = index_transcript(turns)
        assessment = [s for s in statements if s.section == "assessment"]
        icd = check_icd(icd_raw, codeset=self.codeset, grounder=grounder, index=index, assessment=assessment)
        procedures = check_procedures(proc_raw, grounder=grounder)
        caregiver = any(t.speaker == "caregiver" for t in turns)
        mdm = estimate(
            statements, meds, visit_type=visit.visit_type, caregiver_present=caregiver, procedures=len(procedures)
        )
        em = check_em(em_raw, visit_type=visit.visit_type, estimate=mdm)
        transcript_text = " ".join(t.text for t in turns)
        note_text = " ".join(s.text for s in statements) + " " + " ".join(f"{m.name} {m.dose}" for m in meds)
        return Note(
            visit_id=visit.id,
            statements=statements,
            medications=meds,
            allergies=allergies,
            after_visit_summary=avs,
            avs_flags=check_avs(avs, note_text=note_text, transcript_text=transcript_text),
            icd10=icd,
            em=em,
            procedures=procedures,
            mdm=mdm,
        )


def _is_pediatric(visit: Visit) -> bool:
    if visit.specialty == "pediatrics":
        return True
    match = re.match(r"(\d{4})", visit.patient.dob or "")
    year = re.match(r"(\d{4})", visit.visit_date or "")
    return bool(match and year and int(year.group(1)) - int(match.group(1)) < 18)


def regrade(note: Note, visit: Visit, codeset: CodeSet) -> Note:
    """Re-run the guards after a clinician edit (statements, medications), keeping acknowledgements."""
    if visit.transcript is None:
        return note
    turns = visit.transcript.turns
    acknowledged = {(s.id, f.kind) for s in note.statements for f in s.flags if f.acknowledged}
    fresh: list[Statement] = []
    for s in note.statements:
        base = s.model_copy(update={"flags": [], "support": "unsupported"})
        if s.origin == "clinician":
            base.evidence = [e for e in base.evidence if e.verified] or base.evidence
        fresh.append(base)
    statements, meds, _ = Scribe._verify(fresh, note.medications, note.allergies, turns, visit)
    for s in statements:
        if s.origin == "clinician" and s.support == "unsupported":
            # A clinician may document what the recording did not capture: keep it, mark it as their own.
            s.flags = [f for f in s.flags if f.kind != "unsupported"]
            s.flags.append(
                Flag(kind="unsupported", severity="info", message="Added by the clinician; not in the recording.")
            )
        for f in s.flags:
            if (s.id, f.kind) in acknowledged:
                f.acknowledged = True
    note.statements = statements
    note.medications = meds
    return note


def evidence_turns(evidence: Sequence[Evidence]) -> list[int]:
    return sorted({e.turn for e in evidence if e.verified})


__all__ = ["PLACEHOLDER_RE", "DraftResult", "LLMError", "Scribe", "ScribeError", "regrade"]
