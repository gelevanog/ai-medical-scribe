"""How often do the guards catch the errors they exist for? Deliberately corrupt real drafted notes and re-run them.

Corruptions (each applied separately to a copy of a note):
  negation_flip      "Denies chest pain" -> "Reports chest pain" (a statement that matches an absent gold fact)
  number_change      a vital sign or score in an objective statement changed (152 -> 182)
  dose_x10           a medication dose multiplied by ten (20 mg -> 200 mg)
  unit_swap          a dose unit swapped (mg <-> mcg)
  invented           a statement copied from a visit of another specialty, without evidence
  icd_header         a billable ICD-10-CM code replaced by its non-billable category (E11.65 -> E11)
  icd_fake           a code that does not exist (the category with an invented ".9999" extension)
  icd_denied         a code added for a symptom the patient denied (e.g. R07.9 chest pain when chest pain was denied)
A corruption is caught if the corrupted item carries a new error-level flag of the expected kind.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from notewell.codes.icd10 import load_codeset
from notewell.config import Settings
from notewell.dataset import Case, load_cases
from notewell.eval.note_eval import match_fact, statement_polarity
from notewell.guards.coding import check_icd
from notewell.guards.grounding import Grounder
from notewell.guards.negation import index_transcript
from notewell.models import Evidence, Note, Statement
from notewell.scribe.pipeline import Scribe

_DENIED_CODES = {"chest_pain": "R07.9", "dyspnea": "R06.00", "fever": "R50.9", "cough": "R05.9", "headache": "R51.9",
                 "nausea": "R11.0", "vomiting": "R11.10", "palpitations": "R00.2", "syncope": "R55", "edema": "R60.9",
                 "rash": "R21", "abdominal_pain": "R10.9", "diarrhea": "R19.7", "dysuria": "R30.0",
                 "suicidal_ideation": "R45.851", "fatigue": "R53.83", "weight_loss": "R63.4", "back_pain": "M54.50"}  # fmt: skip


def _verify(note: Note, case: Case) -> Note:
    visit = case.visit()
    visit.transcript = case.transcript()
    fresh = [s.model_copy(update={"flags": [], "support": "unsupported"}) for s in note.statements]
    meds = [m.model_copy(update={"flags": []}) for m in note.medications]
    statements, meds, _ = Scribe._verify(fresh, meds, note.allergies, case.dialogue, visit)
    return note.model_copy(update={"statements": statements, "medications": meds})


def _errors(flags: list[Any], kinds: set[str]) -> bool:
    return any(f.kind in kinds and f.severity == "error" for f in flags)


def run_guard_eval(settings: Settings, run: str = "main", seed: int = 7) -> dict[str, Any]:
    rng = random.Random(seed)
    notes_dir = settings.results_dir / "runs" / run / "notes"
    cases = {c.id: c for c in load_cases(settings.dataset_dir)}
    codeset = load_codeset(str(settings.codes_dir))
    notes: dict[str, Note] = {}
    for path in sorted(notes_dir.glob("*.json")):
        notes[path.stem] = Note.model_validate(json.loads(path.read_text(encoding="utf-8"))["note"])
    trials: Counter[str] = Counter()
    caught: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    baseline_errors = 0
    baseline_statements = 0
    other_statements = [(cid, s) for cid, n in notes.items() for s in n.statements]

    def record(kind: str, ok: bool, example: str) -> None:
        trials[kind] += 1
        caught[kind] += ok
        if not ok:
            examples.setdefault(kind, []).append(example[:160])

    for cid, note in notes.items():
        case = cases[cid]
        base = _verify(note, case)
        baseline_statements += len(base.statements)
        baseline_errors += sum(
            1 for s in base.statements if _errors(s.flags, {"negation", "attribution", "hypothetical"})
        )
        # negation flip
        absent = [f for f in case.gold.facts if f.polarity == "absent"]
        for s in base.statements:
            fact = next(
                (f for f in absent if (m := match_fact(s.text, f.match)) and statement_polarity(s.text, m) == "absent"),
                None,
            )
            if fact is None or _errors(s.flags, {"negation"}):
                continue
            flipped = re.sub(r"^(?:denies|no|negative for|without)\s+", "Reports ", s.text, flags=re.I)
            if flipped == s.text:
                flipped = re.sub(r"\b(?:denies|no|not|without)\b\s*", "", s.text, count=1, flags=re.I)
                flipped = "Reports " + flipped[:1].lower() + flipped[1:]
            out = _verify(_replace(base, s.id, flipped), case)
            record("negation_flip", _errors(_stmt(out, s.id).flags, {"negation"}), f"{cid}: {flipped}")
            break
        # number change in objective
        for s in base.statements:
            nums = re.findall(r"\b\d{2,3}\b", s.text)
            if s.section != "objective" or not nums or s.flags:
                continue
            target = nums[0]
            changed = re.sub(rf"\b{target}\b", str(int(target) + 30), s.text, count=1)
            out = _verify(_replace(base, s.id, changed), case)
            record(
                "number_change",
                _errors(_stmt(out, s.id).flags, {"partial_support", "unsupported"}),
                f"{cid}: {changed}",
            )
            break
        # medication dose x10 and unit swap
        for i, med in enumerate(base.medications):
            d = re.match(r"\s*(\d+(?:\.\d+)?)\s*(\w+)", med.dose or "")
            if not d or med.flags:
                continue
            value = float(d.group(1)) * 10
            tampered = base.model_copy(deep=True)
            tampered.medications[i].dose = f"{value:g} {d.group(2)}"
            out = _verify(tampered, case)
            record(
                "dose_x10",
                _errors(out.medications[i].flags, {"dose_range", "dose_mismatch"}),
                f"{cid}: {med.name} {tampered.medications[i].dose}",
            )
            unit = d.group(2).lower()
            swapped_unit = (
                "mcg"
                if unit.startswith(("mg", "milligram"))
                else "mg"
                if unit.startswith(("mcg", "microgram"))
                else None
            )
            if swapped_unit:
                tampered = base.model_copy(deep=True)
                tampered.medications[i].dose = f"{d.group(1)} {swapped_unit}"
                out = _verify(tampered, case)
                record(
                    "unit_swap",
                    _errors(out.medications[i].flags, {"dose_unit", "dose_range"}),
                    f"{cid}: {med.name} {tampered.medications[i].dose}",
                )
            break
        # invented statement from another specialty
        pool = [
            s
            for oid, s in other_statements
            if cases[oid].specialty != case.specialty and s.section in {"objective", "assessment"}
        ]
        if pool:
            alien = rng.choice(pool)
            injected = base.model_copy(deep=True)
            injected.statements.append(Statement(id="x1", section=alien.section, text=alien.text,
                                                 evidence=[Evidence(turn=rng.randrange(len(case.dialogue)), quote=alien.text[:60])]))  # fmt: skip
            out = _verify(injected, case)
            record(
                "invented", _errors(_stmt(out, "x1").flags, {"unsupported", "partial_support"}), f"{cid}: {alien.text}"
            )
        # ICD corruptions
        grounder = Grounder(case.dialogue)
        index = index_transcript(case.dialogue)
        assessment = [s for s in base.statements if s.section == "assessment"]
        primary = next((c for c in note.icd10 if c.valid and len(c.code) > 3), None)
        if primary:
            for kind, code in (("icd_header", primary.code.split(".")[0]), ("icd_fake", primary.code[:3] + ".9999")):
                result = check_icd(
                    [{"code": code}], codeset=codeset, grounder=grounder, index=index, assessment=assessment
                )
                record(kind, _errors(result[0].flags, {"code_invalid", "code_not_billable"}), f"{cid}: {code}")
        denied = next(
            (c for c, ms in index.mentions.items() if index.summary(c) == "absent" and c in _DENIED_CODES), None
        )
        if denied:
            code = _DENIED_CODES[denied]
            result = check_icd([{"code": code}], codeset=codeset, grounder=grounder, index=index, assessment=assessment)
            record("icd_denied", _errors(result[0].flags, {"code_negated"}), f"{cid}: {code} for denied {denied}")
    summary = {
        "run": run,
        "notes": len(notes),
        "catch_rate": {
            k: {"trials": trials[k], "caught": caught[k], "rate": round(caught[k] / trials[k], 3)}
            for k in sorted(trials)
        },
        "overall": {"trials": sum(trials.values()), "caught": sum(caught.values())},
        "baseline_polarity_error_flags_per_statement": round(baseline_errors / max(1, baseline_statements), 4),
        "missed_examples": {k: v[:6] for k, v in examples.items()},
    }
    out_path = settings.results_dir / "guards.json"
    out_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return {k: v for k, v in summary.items() if k != "missed_examples"}


def _replace(note: Note, sid: str, text: str) -> Note:
    copy = note.model_copy(deep=True)
    for s in copy.statements:
        if s.id == sid:
            s.text = text
    return copy


def _stmt(note: Note, sid: str) -> Statement:
    return next(s for s in note.statements if s.id == sid)


def notes_available(path: Path) -> bool:
    return path.exists() and any(path.glob("*.json"))
