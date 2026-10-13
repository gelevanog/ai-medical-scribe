from __future__ import annotations

import re

import pytest

from notewell.dataset import Case
from notewell.eval.phi_eval import evaluate
from notewell.models import Patient, Turn
from notewell.phi.detector import PHIDetector, Registered, dob_matches, parse_date
from notewell.phi.egress import EgressGuard
from notewell.phi.masker import PLACEHOLDER_RE, Masker, Vault, registered_from
from tests.conftest import PATIENT, turns

DET = PHIDetector()
REG = registered_from(PATIENT, "Dr. Owen Hale")


def kinds(text: str, reg: Registered | None = None) -> dict[str, str]:
    return {s.text: s.type for s in DET.detect(text, reg)}


@pytest.mark.parametrize(
    ("text", "value", "kind"),
    [
        ("Call me at 614-555-0187 after five.", "614-555-0187", "PHONE"),
        ("My cell is (937) 555-0148.", "(937) 555-0148", "PHONE"),
        ("The number is 5 5 5 0 1 4 2 if you need it.", "5 5 5 0 1 4 2", "PHONE"),
        ("Email me at ana.ruiz@example.com please.", "ana.ruiz@example.com", "EMAIL"),
        ("It's j underscore reyes at proton mail dot com.", "j underscore reyes at proton mail dot com", "EMAIL"),
        ("The portal is mychart dot lakeshorehealth dot org, I think.", "mychart dot lakeshorehealth dot org", "URL"),
        ("My social is 512-88-0417.", "512-88-0417", "SSN"),
        ("We moved to 4418 Larchmont Avenue, apartment 2B last year.", "4418 Larchmont Avenue", "ADDRESS"),
        ("It's on the 14th of October.", "the 14th of October", "DATE"),
        ("I'm off on the 19th.", "the 19th", "DATE"),
        ("She noticed it back in July, I think.", "July", "DATE"),
        ("Mom is 94 years old now.", "94 years old", "AGE"),
        ("My grandmother turned ninety-three in May.", "ninety-three", "AGE"),
        ("I work at Brightwater Logistics downtown.", "Brightwater Logistics", "ORG"),
        ("He goes to Lincoln Elementary.", "Lincoln Elementary", "ORG"),
    ],
)
def test_patterns(text: str, value: str, kind: str) -> None:
    found = kinds(text)
    assert value in found, found
    assert found[value] == kind


def test_member_id_spelled_out() -> None:
    spans = DET.detect("My member ID is H as in hotel 4 4 7 1 9 0 2.")
    assert any(s.type == "ID" and s.text.endswith("2") and "member" not in s.text for s in spans)


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("My husband Tomasz drove me today.", "Tomasz"),
        ("Dr. Chen bumped it up last spring.", "Chen"),
        ("i saw doctor kowalski at urgent care.", "kowalski"),
        ("My name is Heaven, like the word.", "Heaven"),
        ("That's N-G-U-Y-E-N, with a G.", "N-G-U-Y-E-N"),
        ("Thanks, Priya, that helps.", "Priya"),
        ("Yesterday Zebulon Quarrington came by.", "Zebulon Quarrington"),
        ("My sister Precious lives with me.", "Precious"),
    ],
)
def test_names(text: str, name: str) -> None:
    found = kinds(text)
    assert found.get(name) == "NAME", found


@pytest.mark.parametrize(
    "text",
    [
        "Your blood pressure is 152 over 94 and your pulse is 72.",
        "Take 10 milligrams of lisinopril every morning, and Tylenol as needed.",
        "This looks like Hashimoto's, and we will check a TSH in 2019 style labs.",
        "I will call you next Tuesday about the MRI.",
        "Weight is 214 pounds, temperature 98.6.",
    ],
)
def test_clinical_text_is_not_masked(text: str) -> None:
    assert DET.detect(text) == [], DET.detect(text)


def test_dates_parse_and_dob() -> None:
    assert str(parse_date("March 22nd, 1971")) == "1971-03-22"
    assert str(parse_date("the 22nd of March 1971")) == "1971-03-22"
    assert str(parse_date("3/22/71")) == "1971-03-22"
    assert parse_date("next Tuesday") is None
    assert dob_matches("March 22nd", parse_date("1971-03-22"))
    spans = DET.detect("It's March 22nd, 1971.", REG)
    assert [s.type for s in spans] == ["DOB"]


def test_registered_identifiers_and_asr_misspelling() -> None:
    spans = DET.detect("Will you see Ms. Ferrera about 6145550187 and MRN 4827163?", REG)
    texts = {s.text: (s.type, s.who) for s in spans}
    assert texts["Ferrera"] == ("NAME", "patient")  # fuzzy match of the registered surname
    assert texts["6145550187"][0] == "PHONE"
    assert texts["4827163"][0] == "ID"


def test_registered_common_word_name_only_when_capitalized() -> None:
    reg = registered_from(Patient(name="Will Grace"))
    assert not [s for s in DET.detect("I will try it.", reg) if s.type == "NAME"]
    assert [s for s in DET.detect("Thanks, Will.", reg) if s.type == "NAME"]


def test_masking_links_names_and_restores() -> None:
    t = turns(
        ("clinician", "Can you confirm your name and date of birth?"),
        ("patient", "Marisol Ferreira, March 22nd, 1971. My husband Tomasz Nowak is here."),
        ("clinician", "Thanks, Marisol. Hi Tomasz, I'm Dr. Hale."),
        ("patient", "Tomasz has my phone, 614-555-0187."),
    )
    masked = Masker().mask_turns(t, REG)
    text = masked.text()
    for secret in ("Marisol", "Ferreira", "1971", "Tomasz", "Nowak", "Hale", "555-0187"):
        assert secret not in text
    assert "[PATIENT]" in text and "[DOB]" in text
    placeholders = set(PLACEHOLDER_RE.findall(text))
    tomasz = next(p for p, v in masked.vault.originals.items() if "Tomasz" in v)
    assert text.count(tomasz) == 3  # "Tomasz Nowak" and both "Tomasz" are one person
    assert masked.vault.restore(f"{tomasz} drove [PATIENT].") == "Tomasz Nowak drove Marisol Ferreira."
    assert len(placeholders) >= 4


def test_vault_roundtrip_and_unknown_placeholders() -> None:
    masked = Masker().mask_turns(turns(("patient", "I'm Marisol Ferreira, call 614-555-0187.")), REG)
    data = masked.vault.to_dict()
    vault = Vault.from_dict(data)
    assert vault.restore("[PATIENT] at [PHONE_1]") == "Marisol Ferreira at 614-555-0187"
    assert vault.unknown_placeholders("[NAME_9] and [PATIENT]") == ["[NAME_9]"]


def test_propagation_masks_every_mention() -> None:
    t = turns(("patient", "My neighbor Odalys helps me."), ("patient", "odalys picks up the pills, Odalys is great."))
    masked = Masker().mask_turns(t, Registered())
    assert "dalys" not in masked.text()


def test_egress_guard_finds_what_the_masker_missed() -> None:
    masked = Masker().mask_turns(turns(("patient", "I'm Marisol Ferreira.")), REG)
    guard = EgressGuard(REG, masked.vault)
    assert guard.scan([{"role": "user", "content": masked.text()}]).clean
    dirty = guard.scan([{"role": "user", "content": "Patient Ferreirra, phone 614 555 0187, born 3/22/1971"}])
    assert not dirty.clean
    assert {"NAME_FUZZY", "PHONE", "DOB"} <= set(dirty.hits)
    assert "Ferreirra" in dirty.found
    with pytest.raises(Exception, match="egress check"):
        guard.check([{"role": "user", "content": "Marisol says hi"}])


def test_egress_guard_ignores_common_words_for_word_names() -> None:
    guard = EgressGuard(registered_from(Patient(name="Will Hope")))
    assert guard.scan([{"role": "user", "content": "I will hope for the best."}]).clean
    assert not guard.scan([{"role": "user", "content": "Will Hope is here."}]).clean


def test_no_registered_identifier_survives_masking_in_any_visit(cases: list[Case]) -> None:
    masker = Masker()
    for case in cases:
        reg = registered_from(case.patient, case.clinician.name)
        masked = masker.mask_turns(case.dialogue, reg)
        guard = EgressGuard(reg, masked.vault)
        report = guard.scan([{"role": "user", "content": masked.text()}])
        assert report.clean, (case.id, report.summary())


def test_dataset_gold_identifiers_recall(cases: list[Case]) -> None:
    result = evaluate(cases)
    assert result.recall >= 0.98, result.misses
    assert result.masked_tokens_not_phi / result.masked_tokens < 0.08


def test_masked_text_keeps_titles_and_clinical_content() -> None:
    masked = Masker().mask_turns(
        turns(("clinician", "Dr. Chen started lisinopril 10 milligrams on the 3rd of May.")), Registered()
    )
    text = masked.turns[0].text
    assert text.startswith("Dr. [NAME_1] started lisinopril 10 milligrams on [DATE_1]")
    assert re.search(r"\[DATE_\d\]", text)


def test_turn_model_roundtrip() -> None:
    t = Turn(idx=0, speaker="patient", text="hello")
    assert Turn.model_validate(t.model_dump()) == t
