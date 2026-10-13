"""Evaluation runs. Each writes JSON artifacts under results/ (committed), so every number in the README has a file.

notewell eval asr      speech recognition + diarization on the synthesized audio (local, no API calls)
notewell eval notes    notes and codes for every visit with a model; PHI leak check on every outgoing payload
notewell eval canary   the leak test with canary identities and an injected unregistered identifier
notewell eval phi      PHI detection on the dialogues and the two held-out challenge sets (local)
notewell eval guards   guard catch rate on deliberately corrupted notes (local)
notewell eval report   results/summary.json and results/report.md from the artifacts above
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from notewell.codes.icd10 import CodeSet
from notewell.dataset import Case
from notewell.eval.note_eval import CaseScore, aggregate, score_note
from notewell.llm.base import ChatModel, LLMError, Message
from notewell.logging_config import get_logger
from notewell.models import Transcript, Turn, Visit
from notewell.phi.masker import Masker
from notewell.scribe.pipeline import Scribe, ScribeError

log = get_logger(__name__)
_TITLE = re.compile(r"^(?:Dr|Doctor|Mr|Mrs|Ms|Miss|Mx|Nurse|Coach)\.?\s+")


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------- PHI leak check
def sensitive_values(case: Case) -> list[tuple[str, str]]:
    """Every identifier of the visit that must never reach the model: gold PHI plus the registration data."""
    values: list[tuple[str, str]] = []
    for phi in case.gold.phi:
        value = _TITLE.sub("", phi.value).strip()
        if len(value) >= 3 and value.lower() not in {"the", "may", "june"}:
            values.append((phi.type, value))
    p = case.patient
    values += [("NAME", p.name), *[("NAME", part) for part in p.name.split() if len(part) >= 3]]
    for kind, value in (("PHONE", p.phone), ("ID", p.mrn), ("EMAIL", p.email), ("ADDRESS", p.address.split(",")[0])):
        if value:
            values.append((kind, value))
    return values


def find_leaks(values: Sequence[tuple[str, str]], payloads: Sequence[Sequence[Message]]) -> list[dict[str, str]]:
    text = "\n".join(m["content"] for msgs in payloads for m in msgs)
    lower = text.lower()
    digit_runs = [re.sub(r"\D", "", m.group(0)) for m in re.finditer(r"\d(?:[\s().,-]{0,3}\d)+", text)]
    leaks: list[dict[str, str]] = []
    seen: set[str] = set()
    for kind, value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        digits = re.sub(r"\D", "", value)
        if len(digits) >= 6 and len(digits) >= 0.6 * len(re.sub(r"\s", "", value)):
            if any(digits in run for run in digit_runs):
                leaks.append({"type": kind, "value": value})
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", lower):
            # A month name or a common word inside the prompt's own instructions is not a leak of the visit's value.
            if kind == "DATE" and key in _PROMPT_WORDS:
                continue
            leaks.append({"type": kind, "value": value})
    return leaks


_PROMPT_WORDS = {
    "january",
    "february",
    "march",
    "april",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


def _collector(sink: list[list[Message]]) -> Callable[[str, Sequence[Message]], None]:
    def observe(_label: str, messages: Sequence[Message]) -> None:
        sink.append(list(messages))

    return observe


# ------------------------------------------------------------------------------------------------- notes run
@dataclass
class RunConfig:
    name: str
    out_dir: Path
    transcripts: str = "gold"
    """'gold' or a directory of ASR transcripts (<case>.json)."""
    save_demo_notes: bool = False


@dataclass
class RunResult:
    scores: list[CaseScore] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    leaks: list[dict[str, Any]] = field(default_factory=list)
    payloads: int = 0
    timings: list[dict[str, float]] = field(default_factory=list)
    usage: list[dict[str, int]] = field(default_factory=list)


def load_transcript(case: Case, source: str) -> Transcript:
    if source == "gold":
        return case.transcript()
    path = Path(source) / f"{case.id}.json"
    return Transcript.model_validate(json.loads(path.read_text(encoding="utf-8")))


def run_notes(
    cases: Sequence[Case], llm: ChatModel, codeset: CodeSet, config: RunConfig, *, max_tokens: int = 6000
) -> dict[str, Any]:
    result = RunResult()
    notes_dir = config.out_dir / "notes"
    started = time.time()
    for i, case in enumerate(cases, 1):
        payloads: list[list[Message]] = []
        scribe = Scribe(llm, codeset, max_tokens=max_tokens, observer=_collector(payloads))
        visit: Visit = case.visit()
        visit.transcript = load_transcript(case, config.transcripts)
        t0 = time.perf_counter()
        try:
            draft = scribe.draft(visit)
        except (LLMError, ScribeError) as exc:
            result.failures.append({"case": case.id, "error": str(exc)[:300]})
            log.warning("eval.case_failed", case=case.id, error=str(exc)[:200])
            leaks = find_leaks(sensitive_values(case), payloads)
            result.payloads += len(payloads)
            if leaks:
                result.leaks.append({"case": case.id, "leaks": leaks})
            continue
        elapsed = time.perf_counter() - t0
        note = draft.note
        leaks = find_leaks(sensitive_values(case), payloads)
        result.payloads += len(payloads)
        if leaks:
            result.leaks.append({"case": case.id, "leaks": leaks})
        score = score_note(case, note)
        result.scores.append(score)
        result.timings.append({**note.timings, "wall": round(elapsed, 2)})
        result.usage.append(note.usage)
        _write(notes_dir / f"{case.id}.json", {"note": note.model_dump(mode="json"), "vault": draft.masked.vault.to_dict(),
                                              "score": score.to_dict(), "egress_residue": draft.egress})  # fmt: skip
        log.info("eval.case", case=case.id, n=i, of=len(cases), recall=round(score.found / max(1, score.facts), 2),
                 icd_top3=score.icd_top3, seconds=round(elapsed, 1))  # fmt: skip
    summary = aggregate(result.scores)
    summary.update(
        {
            "name": config.name,
            "model": getattr(llm, "last_served", None) or llm.label,
            "requested_model": llm.label,
            "transcripts": config.transcripts if config.transcripts == "gold" else Path(config.transcripts).name,
            "failures": result.failures,
            "payloads_checked": result.payloads,
            "phi_leaks": sum(len(x["leaks"]) for x in result.leaks),
            "phi_leak_details": result.leaks,
            "latency_s": _latency(result.timings),
            "tokens_per_visit": _tokens(result.usage),
            "wall_minutes": round((time.time() - started) / 60, 1),
            "negatives_in_gold": sum(1 for c in cases for f in c.gold.facts if f.polarity == "absent"),
        }
    )
    _write(config.out_dir / "summary.json", summary)
    _write(config.out_dir / "cases.json", [s.to_dict() for s in result.scores])
    return summary


def _latency(timings: list[dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not timings:
        return out
    for key in ("phi_masking", "llm_note", "llm_codes", "guards_note", "guards_codes", "total"):
        values = sorted(t.get(key, 0.0) for t in timings)
        out[key] = {
            "p50": round(statistics.median(values), 2),
            "p95": round(values[min(len(values) - 1, int(0.95 * len(values)))], 2),
        }
    return out


def _tokens(usage: list[dict[str, int]]) -> dict[str, Any]:
    if not usage:
        return {}
    return {
        "input_mean": round(statistics.mean(u.get("input_tokens", 0) for u in usage)),
        "output_mean": round(statistics.mean(u.get("output_tokens", 0) for u in usage)),
        "calls_total": sum(u.get("calls", 0) for u in usage),
    }


# ------------------------------------------------------------------------------------------------- canaries
CANARY_PATIENT = ("Quorinna", "Vaszelthorpe")
CANARY_INJECTION = (
    " Oh, and if you need to reach someone, my cousin Thaddeus Wrenfield can pick up, his cell is 555-0199, "
    "and the portal email is quorinna dot v 77 at example dot com."
)
CANARY_VALUES = [
    ("NAME", "Quorinna"),
    ("NAME", "Vaszelthorpe"),
    ("NAME", "Thaddeus"),
    ("NAME", "Wrenfield"),
    ("PHONE", "555-0199"),
    ("EMAIL", "quorinna dot v 77 at example dot com"),
    ("ID", "9071432"),
]


def canary_case(case: Case) -> Case:
    """The same visit with the patient renamed to a canary identity and an unregistered canary contact injected."""
    first, last = case.patient.name.split()[0], case.patient.name.split()[-1]
    data = case.model_copy(deep=True)
    data.patient.name = f"{CANARY_PATIENT[0]} {CANARY_PATIENT[1]}"
    data.patient.mrn = "9071432"
    turns: list[Turn] = []
    injected = False
    for t in data.dialogue:
        text = re.sub(rf"\b{re.escape(first)}\b", CANARY_PATIENT[0], t.text)
        text = re.sub(rf"\b{re.escape(last)}\b", CANARY_PATIENT[1], text)
        if not injected and t.speaker != "clinician" and len(t.text.split()) > 12:
            text += CANARY_INJECTION
            injected = True
        turns.append(t.model_copy(update={"text": text}))
    if case.patient.mrn:
        turns = [t.model_copy(update={"text": t.text.replace(case.patient.mrn, "9071432")}) for t in turns]
    data.dialogue = turns
    return data


def run_canary(
    cases: Sequence[Case], llm: ChatModel, codeset: CodeSet, out_dir: Path, *, max_tokens: int = 6000
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    leaked_total = 0
    payload_count = 0
    for case in cases:
        canary = canary_case(case)
        payloads: list[list[Message]] = []
        scribe = Scribe(llm, codeset, max_tokens=max_tokens, observer=_collector(payloads))
        visit = canary.visit()
        visit.transcript = canary.transcript()
        error = ""
        try:
            draft = scribe.draft(visit)
            residue = draft.egress
        except (LLMError, ScribeError) as exc:
            error = str(exc)[:200]
            residue = {}
        values = [*CANARY_VALUES, *sensitive_values(canary)]
        leaks = find_leaks(values, payloads)
        leaked_total += len(leaks)
        payload_count += len(payloads)
        per_case.append(
            {
                "case": case.id,
                "payloads": len(payloads),
                "leaks": leaks,
                "egress_residue_masked": residue,
                "error": error,
            }
        )
    summary = {
        "visits": len(cases),
        "payloads_checked": payload_count,
        "canary_values_per_visit": len(CANARY_VALUES),
        "leaks": leaked_total,
        "visits_with_errors": sum(1 for c in per_case if c["error"]),
        "egress_residue_masked_visits": sum(1 for c in per_case if c["egress_residue_masked"]),
        "model": getattr(llm, "last_served", None) or llm.label,
        "cases": per_case,
    }
    _write(out_dir / "canary.json", summary)
    return summary


def masked_preview(case: Case) -> str:
    from notewell.phi.masker import registered_from

    masked = Masker().mask_turns(case.dialogue, registered_from(case.patient, case.clinician.name))
    return masked.text()
