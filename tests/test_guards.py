from __future__ import annotations

import pytest

from notewell.codes.em import estimate
from notewell.guards.avs import check_avs, reading_grade
from notewell.guards.coding import check_em, check_icd, check_procedures
from notewell.guards.grounding import Grounder
from notewell.guards.medications import check_medication, normalize_drug, parse_dose, times_per_day, weight_kg
from notewell.guards.negation import check_statement, index_transcript
from notewell.guards.text import numbers, words_to_digits
from notewell.models import Allergy, Evidence, Medication, Statement
from tests.conftest import turns

T = turns(
    ("clinician", "What brings you in today?"),
    ("patient", "I've been getting headaches for about two weeks, mostly in the morning."),
    ("clinician", "Any chest pain or shortness of breath?"),
    ("patient", "No, nothing like that."),
    ("clinician", "Are you still taking the lisinopril?"),
    ("patient", "Yes, 10 milligrams every morning. Wait, no, it's 20 now."),
    ("patient", "My dad had a heart attack at 50 and my mom has diabetes."),
    ("clinician", "Your blood pressure today is 152 over 94, pulse 72. Lungs are clear."),
    ("clinician", "If you get chest pain or a severe headache, go to the emergency room."),
    ("clinician", "I'd like to increase the lisinopril to 40 milligrams once a day. We'll recheck in 6 weeks."),
)


def stmt(text: str, section: str = "subjective", turn: int | None = None, quote: str = "") -> Statement:
    ev = [Evidence(turn=turn, quote=quote)] if turn is not None else []
    return Statement(id="s1", section=section, text=text, evidence=ev)  # type: ignore[arg-type]


def test_number_words() -> None:
    assert words_to_digits("two weeks and twenty-five days") == "2 weeks and 25 days"
    assert words_to_digits("7 and a half milliliters") == "7.5 milliliters"
    assert numbers("BP 152/94, 1,440 mg") == {"152", "94", "1440"}
    assert numbers("in one day", words=False) == set()


def test_grounding_supported_with_verified_quote() -> None:
    g = Grounder(T).ground(
        stmt("Headaches for 2 weeks, mostly in the morning", turn=1, quote="getting headaches for about two weeks")
    )
    assert g.support == "supported" and g.evidence[0].verified and not g.flags


def test_grounding_fixes_a_wrong_turn_number() -> None:
    g = Grounder(T).ground(stmt("Pulse 72", "objective", turn=3, quote="pulse 72"))
    assert g.support == "supported" and g.evidence[0].turn == 7


def test_grounding_flags_a_number_nobody_said() -> None:
    g = Grounder(T).ground(
        stmt("Blood pressure 182 over 94", "objective", turn=7, quote="Your blood pressure today is 152 over 94")
    )
    assert g.support == "partial"
    assert g.flags[0].severity == "error" and "182" in g.flags[0].message


def test_grounding_unsupported_without_evidence() -> None:
    g = Grounder(T).ground(stmt("Patient reports a rash on both arms for 3 days"))
    assert g.support == "unsupported" and g.flags[0].kind == "unsupported"


def test_grounding_autolinks_when_the_model_gave_no_quote() -> None:
    g = Grounder(T).ground(stmt("Increase lisinopril to 40 mg once daily", "plan"))
    assert g.support == "supported" and g.evidence[0].turn == 9


def test_negation_question_answer() -> None:
    index = index_transcript(T)
    assert index.summary("chest_pain") == "absent"
    assert index.summary("dyspnea") == "absent"
    assert [i.kind for i in check_statement("Reports chest pain with exertion", index)] == ["negation"]
    assert check_statement("Denies chest pain", index) == []


def test_negation_ignores_precautions_and_temporal_statements() -> None:
    index = index_transcript(T)
    assert check_statement("Go to the ER for chest pain", index, section="plan") == []
    assert check_statement("Return if chest pain develops", index) == []


def test_denying_something_described_is_flagged() -> None:
    index = index_transcript(T)
    issues = check_statement("Denies headaches", index)
    assert issues and issues[0].kind == "negation"


def test_family_history_attribution() -> None:
    index = index_transcript(T)
    assert index.summary("diabetes") == "unknown"  # not a symptom concept in the lexicon
    t2 = turns(("patient", "My dad had chest pain for years before his bypass."))
    idx2 = index_transcript(t2)
    assert idx2.summary("chest_pain") == "family"
    assert [i.kind for i in check_statement("Chest pain for years", idx2)] == ["attribution"]
    assert check_statement("Family history: father had chest pain", idx2) == []


def test_hypothetical_only_mention() -> None:
    t2 = turns(("clinician", "If the rash spreads to your face, come back."))
    idx = index_transcript(t2)
    assert idx.summary("rash") == "hypothetical"
    assert [i.kind for i in check_statement("Rash on the face", idx)] == ["hypothetical"]


def test_caregiver_talking_about_the_child_is_the_patient() -> None:
    t2 = turns(("clinician", "What's going on?"), ("caregiver", "My daughter has had a fever since yesterday."))
    assert index_transcript(t2).summary("fever") == "present"


def test_dose_parsing_and_frequency() -> None:
    assert parse_dose("88 micrograms") == (88.0, "mcg", False)
    assert parse_dose("90 mg/kg/day") == (90.0, "mg", True)
    assert times_per_day("twice a day") == 2
    assert times_per_day("every 6 hours") == 4
    assert times_per_day("at bedtime") == 1
    assert normalize_drug("Eliquis").name == "apixaban"  # type: ignore[union-attr]
    assert normalize_drug("unobtainium") is None


def kinds(flags: list) -> list[str]:  # type: ignore[type-arg]
    return [f.kind for f in flags]


def test_medication_checks() -> None:
    ok = check_medication(Medication(name="lisinopril", dose="40 mg", frequency="once daily"), turns=T, allergies=[])
    assert "dose_range" not in kinds(ok) and "dose_mismatch" not in [f.kind for f in ok if f.severity == "error"]
    high = check_medication(Medication(name="lisinopril", dose="400 mg", frequency="once daily"), turns=T, allergies=[])
    assert "dose_range" in kinds(high) and "dose_mismatch" in kinds(high)
    unit = check_medication(Medication(name="levothyroxine", dose="88 mg", frequency="daily"), turns=T, allergies=[])
    assert "dose_unit" in kinds(unit)
    daily = check_medication(
        Medication(name="ibuprofen", dose="800 mg", frequency="every 4 hours"), turns=T, allergies=[]
    )
    assert "dose_range" in kinds(daily)
    unknown = check_medication(Medication(name="zorbex", dose="5 mg"), turns=T, allergies=[])
    assert kinds(unknown)[0] == "unknown_drug"


def test_dose_correction_in_conversation_is_surfaced() -> None:
    flags = check_medication(Medication(name="lisinopril", dose="20 mg", frequency="daily"), turns=T, allergies=[])
    corr = [f for f in flags if f.kind == "dose_mismatch" and f.severity == "warning"]
    assert corr and "corrected" in corr[0].message


def test_allergy_conflicts() -> None:
    pen = [Allergy(substance="penicillin", reaction="hives")]
    assert "allergy_conflict" in kinds(
        check_medication(
            Medication(name="amoxicillin", dose="500 mg", frequency="three times a day"), turns=[], allergies=pen
        )
    )
    ceph = check_medication(
        Medication(name="cephalexin", dose="500 mg", frequency="twice a day"), turns=[], allergies=pen
    )
    assert ceph[-1].kind == "allergy_conflict" and ceph[-1].severity == "warning"
    sulfa = [Allergy(substance="sulfa")]
    assert "allergy_conflict" in kinds(
        check_medication(Medication(name="Bactrim", dose="800 mg"), turns=[], allergies=sulfa)
    )


def test_pediatric_weight_based_dose() -> None:
    t = turns(
        ("clinician", "Her weight today is 16 kilograms."), ("clinician", "Amoxicillin 720 milligrams twice a day.")
    )
    assert weight_kg(t) == 16
    ok = check_medication(
        Medication(name="amoxicillin", dose="720 mg", frequency="twice daily"), turns=t, allergies=[], pediatric=True
    )
    assert "dose_range" not in kinds(ok)
    bad = check_medication(
        Medication(name="amoxicillin", dose="72 mg", frequency="twice daily"), turns=t, allergies=[], pediatric=True
    )
    assert "dose_range" in kinds(bad)


def test_icd_checks(codeset) -> None:  # type: ignore[no-untyped-def]
    g = Grounder(T)
    index = index_transcript(T)
    assessment = [stmt("Essential hypertension, not at goal", "assessment")]
    raw = [{"code": "I10", "evidence": [{"turn": 7, "quote": "blood pressure today is 152 over 94"}]}, {"code": "E11"},
           {"code": "Q99.99"}, {"code": "R07.9"}, {"code": "S39.012A"}, {"code": "I10"}]  # fmt: skip
    out = check_icd(raw, codeset=codeset, grounder=g, index=index, assessment=assessment)
    by = {c.code: c for c in out}
    assert by["I10"].valid and not [f for f in by["I10"].flags if f.severity == "error"]
    assert "code_not_billable" in kinds(by["E11"].flags) and by["E11"].flags[0].detail["children"]
    assert "code_invalid" in kinds(by["Q99.99"].flags)
    assert "code_negated" in kinds(by["R07.9"].flags)  # chest pain was denied
    assert by["S39.012A"].flags[0].severity == "warning"  # outside the offline subset: unverifiable, not invalid
    assert len(out) == 5  # the duplicate is dropped


def test_em_checks() -> None:
    statements = [stmt("Hypertension, not at goal on lisinopril", "assessment"), stmt("Increase lisinopril to 40 mg", "plan"),
                  stmt("Repeat potassium and kidney labs in 2 weeks", "plan")]  # fmt: skip
    meds = [Medication(name="lisinopril", dose="40 mg", status="changed")]
    est = estimate(statements, meds, visit_type="established")
    assert est.code == "99214" and est.risk == "moderate"
    assert check_em({"code": "99214"}, visit_type="established", estimate=est).flags == []
    wrong_family = check_em({"code": "99204"}, visit_type="established", estimate=est)
    assert "em_visit_type" in kinds(wrong_family.flags)
    lower = check_em({"code": "99212"}, visit_type="established", estimate=est)
    assert "em_mismatch" in kinds(lower.flags)
    assert check_em({"code": "90791"}, visit_type="new", estimate=est).flags == []
    assert estimate([], [], visit_type="new").code == "99202"


def test_procedure_checks() -> None:
    out = check_procedures(
        [{"code": "93000", "evidence": [{"turn": 7, "quote": "Lungs are clear"}]}, {"code": "99999"}],
        grounder=Grounder(T),
    )
    assert out[0].valid and "code_invalid" in kinds(out[1].flags)


def test_after_visit_summary_checks() -> None:
    note_text = "Increase lisinopril to 40 mg once daily. Recheck in 6 weeks."
    good = "Your blood pressure is high. Take lisinopril 40 mg once a day. Come back in 6 weeks."
    assert check_avs(good, note_text=note_text, transcript_text="") == []
    bad = check_avs("Take lisinopril 80 mg and metformin daily.", note_text=note_text, transcript_text="")
    assert len([f for f in bad if f.kind == "unsupported"]) == 2
    assert reading_grade("The cat sat. The dog ran.") < 3
    assert check_avs("", note_text="", transcript_text="")[0].kind == "readability"


@pytest.mark.parametrize("text", ["No fever, chills, or cough", "Denies fever and chills", "Negative for fever"])
def test_statement_negation_scope(text: str) -> None:
    t2 = turns(("patient", "I had a fever and chills last night."))
    issues = check_statement(text, index_transcript(t2))
    assert issues and issues[0].kind == "negation"
