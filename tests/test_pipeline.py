from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from notewell.codes.icd10 import CodeSet
from notewell.dataset import Case
from notewell.eval.runner import find_leaks, sensitive_values
from notewell.llm.base import Completion, Message, PolicyViolationError
from notewell.llm.fake import FakeScribe
from notewell.llm.resilient import ResilientModel
from notewell.scribe.parse import ParseError, extract_json, parse_codes, parse_note
from notewell.scribe.pipeline import Scribe, ScribeError


def test_extract_json_tolerates_fences_and_trailing_commas() -> None:
    assert extract_json('```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}
    assert extract_json('Sure! {"x": 1} hope that helps') == {"x": 1}
    with pytest.raises(ParseError):
        extract_json("no json here")


def test_parse_note_shapes() -> None:
    data = {
        "Subjective": [{"text": "Headache for 2 weeks", "evidence": [{"turn": "4", "quote": "headaches"}]}],
        "objective": {"vitals": ["BP 152/94"]},
        "assessment": "Hypertension. Headache.",
        "plan": [],
        "medications": [{"name": "Lisinopril", "dose": "20 mg", "turn": 7, "quote": "20 now"}],
        "allergies": [{"substance": "none"}, {"substance": "penicillin", "reaction": "hives"}],
        "after_visit_summary": "Take care.",
    }
    statements, meds, allergies, avs = parse_note(data)
    assert [s.section for s in statements] == ["subjective", "objective", "assessment", "assessment"]
    assert statements[0].evidence[0].turn == 4
    assert meds[0].evidence[0].turn == 7 and meds[0].status == "active"
    assert [a.substance for a in allergies] == ["penicillin"] and avs == "Take care."
    icd, em, procs = parse_codes({"icd10": ["I10", {"code": "R51.9"}], "em": "99214", "procedures": [{"code": "93000"}]})
    assert [c["code"] for c in icd] == ["I10", "R51.9"] and em == {"code": "99214"} and procs[0]["code"] == "93000"


def _visit(case: Case):  # type: ignore[no-untyped-def]
    v = case.visit()
    v.transcript = case.transcript()
    return v


def test_fake_draft_end_to_end_without_phi_in_any_payload(cases: list[Case], codeset: CodeSet) -> None:
    for case in cases[::4]:
        payloads: list[Sequence[Message]] = []
        scribe = Scribe(ResilientModel(FakeScribe()), codeset, observer=lambda label, m, sink=payloads: sink.append(m))
        result = scribe.draft(_visit(case))
        note = result.note
        assert note.statements and note.em is not None and note.mdm is not None
        assert len(payloads) == 2
        assert find_leaks(sensitive_values(case), payloads) == [], case.id
        assert all(s.support in {"supported", "partial", "unsupported"} for s in note.statements)
        assert note.timings["total"] > 0 and note.phi.get("NAME", 0) >= 1


class Echo:
    """A model that answers with a statement containing a placeholder and one unknown placeholder."""

    label = "test/echo"
    is_local = True

    def complete(self, messages: Sequence[Message], *, max_tokens: int, temperature: float = 0.0) -> Completion:
        if "medical coder" in messages[0]["content"]:
            return Completion(text=json.dumps({"icd10": [{"code": "I10"}], "em": {"code": "99213"}, "procedures": []}), model="echo")
        body = {
            "subjective": [{"text": "[PATIENT] reports headaches for 2 weeks", "evidence": [{"turn": 1, "quote": "headaches for about two weeks"}]}],
            "objective": [], "assessment": [{"text": "Hypertension"}], "plan": [{"text": "Call [NAME_7] tomorrow"}],
            "medications": [], "allergies": [], "after_visit_summary": "See you soon, [PATIENT].",
        }  # fmt: skip
        return Completion(text=json.dumps(body), model="echo")


def test_placeholders_are_restored_locally(case_by_id: dict[str, Case], codeset: CodeSet) -> None:
    case = case_by_id["pc-01"]
    note = Scribe(Echo(), codeset).draft(_visit(case)).note
    assert note.statements[0].text.startswith(case.patient.name)
    assert note.after_visit_summary.endswith(f"{case.patient.name}.")
    assert any(f.kind == "placeholder" for f in note.avs_flags)  # [NAME_7] does not exist


class Leaky:
    """Returns JSON that quotes the patient's name, so the second (codes) payload would carry it unless re-masked."""

    label = "test/leaky"
    is_local = True

    def __init__(self, name: str) -> None:
        self.name = name

    def complete(self, messages: Sequence[Message], *, max_tokens: int, temperature: float = 0.0) -> Completion:
        if "medical coder" in messages[0]["content"]:
            return Completion(text='{"icd10": [], "em": null, "procedures": []}', model="leaky")
        return Completion(text=json.dumps({"assessment": [{"text": f"{self.name} has hypertension"}]}), model="leaky")


def test_note_text_is_masked_again_before_the_codes_call(case_by_id: dict[str, Case], codeset: CodeSet) -> None:
    case = case_by_id["pc-01"]
    payloads: list[Sequence[Message]] = []
    Scribe(Leaky(case.patient.name), codeset, observer=lambda label, m: payloads.append(m)).draft(_visit(case))
    assert find_leaks(sensitive_values(case), payloads) == []


def test_egress_residue_is_masked_and_counted(case_by_id: dict[str, Case], codeset: CodeSet) -> None:
    case = case_by_id["pc-01"]
    visit = _visit(case)
    # A value the detector does not know, registered only as the patient's (fictional) e-mail.
    visit.patient.email = "zq-ledger-77"
    visit.transcript.turns[3].text += " My portal login is zq-ledger-77."  # type: ignore[union-attr]
    result = Scribe(ResilientModel(FakeScribe()), codeset).draft(visit)
    assert result.note.phi.get("NAME", 0) >= 1


def test_block_when_the_egress_check_cannot_be_satisfied(case_by_id: dict[str, Case], codeset: CodeSet, monkeypatch: pytest.MonkeyPatch) -> None:
    from notewell.phi import egress

    case = case_by_id["pc-01"]
    monkeypatch.setattr(egress.EgressReport, "clean", property(lambda self: False))
    with pytest.raises(PolicyViolationError):
        Scribe(ResilientModel(FakeScribe()), codeset).draft(_visit(case))


def test_no_transcript_is_an_error(case_by_id: dict[str, Case], codeset: CodeSet) -> None:
    with pytest.raises(ScribeError):
        Scribe(FakeScribe(), codeset).draft(case_by_id["pc-01"].visit())


class Garbage:
    label = "test/garbage"
    is_local = True
    calls = 0

    def complete(self, messages: Sequence[Message], *, max_tokens: int, temperature: float = 0.0) -> Completion:
        Garbage.calls += 1
        return Completion(text="I cannot produce JSON today.", model="garbage")


def test_one_repair_then_a_clear_error(case_by_id: dict[str, Case], codeset: CodeSet) -> None:
    with pytest.raises(ScribeError, match="after one repair"):
        Scribe(Garbage(), codeset).draft(_visit(case_by_id["pc-01"]))
    assert Garbage.calls == 2
