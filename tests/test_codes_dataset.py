from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from notewell.codes import icd10
from notewell.codes.icd10 import CodeSet, dotted, undotted
from notewell.dataset import SPECIALTIES, Case
from notewell.dataset.validate import validate_case
from notewell.kb import concepts, cpt, drug_aliases, drugs, find_concepts


def test_dotted_forms() -> None:
    assert dotted("e119") == "E11.9"
    assert dotted("I10") == "I10"
    assert undotted("S39.012A") == "S39012A"


def test_subset_validation(codeset: CodeSet) -> None:
    assert not codeset.complete and len(codeset.codes) > 5000
    assert codeset.validate("I10").status == "valid"
    assert codeset.validate("E11.9").title == "Type 2 diabetes mellitus without complications"
    header = codeset.validate("E11")
    assert header.status == "not_billable" and header.children and all(c.startswith("E11.") for c in header.children)
    assert codeset.validate("E99.99").status == "invalid"
    assert codeset.validate("S39.012A").status == "unverifiable"
    assert codeset.validate("hello").status == "invalid"


def test_full_list_validation(full_codeset: CodeSet) -> None:
    assert full_codeset.validate("S39.012A").status == "valid"
    assert full_codeset.validate("S39.012").status == "not_billable"
    assert len(full_codeset.codes) > 90_000


def test_search_ranks_the_obvious_code_first(codeset: CodeSet) -> None:
    assert codeset.search("essential hypertension", limit=3)[0][0].dotted == "I10"
    top = [c.dotted for c, _ in codeset.search("generalized anxiety disorder", limit=3)]
    assert "F41.1" in top
    assert all(c.billable for c, _ in codeset.search("type 2 diabetes", limit=10))


def test_download_verifies_the_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr(
            "x/icd10cm-order-2027.txt",
            "00001 I10     1 Essential (primary) hypertension" + " " * 30 + "Essential (primary) hypertension\n",
        )
    payload = buffer.getvalue()

    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *a: object) -> None:
            return None

    monkeypatch.setattr(icd10.urllib.request, "urlopen", lambda url, timeout: Response(payload))
    with pytest.raises(RuntimeError, match="checksum"):
        icd10.download_full(tmp_path)
    path = icd10.download_full(tmp_path, sha256=hashlib.sha256(payload).hexdigest())
    assert path.read_text().startswith("00001 I10")


def test_drug_list_and_concepts() -> None:
    assert len(drugs()) >= 150
    assert drug_aliases()["eliquis"] == "apixaban"
    assert all(d.dose[0] <= d.dose[1] <= d.max_daily for d in drugs().values())
    assert len(concepts()) >= 80
    assert [c for c, _, _ in find_concepts("Denies chest pain or shortness of breath")] == ["chest_pain", "dyspnea"]


def test_cpt_list_has_no_ama_descriptors() -> None:
    catalog = cpt()
    assert {"99213", "99214", "99204", "93000", "96127", "90791"} <= set(catalog)
    assert catalog["99214"]["group"] == "em_established"


def test_dataset_size_and_spread(cases: list[Case]) -> None:
    assert len(cases) >= 30
    assert {c.specialty for c in cases} == set(SPECIALTIES)
    assert sum(len(c.gold.facts) for c in cases) > 700
    assert all(c.patient.name in " ".join(t.text for t in c.dialogue) for c in cases)
    assert sum(1 for c in cases for f in c.gold.facts if f.polarity == "absent") > 100


def test_every_case_is_consistent(cases: list[Case], codeset: CodeSet) -> None:
    problems = {c.id: validate_case(c, codeset) for c in cases}
    assert not {k: v for k, v in problems.items() if v}


def test_every_gold_code_is_valid_in_the_full_list(cases: list[Case], full_codeset: CodeSet) -> None:
    for case in cases:
        for dx in case.gold.diagnoses:
            assert full_codeset.validate(dx.code).status == "valid", (case.id, dx.code)
