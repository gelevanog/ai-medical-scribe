from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from notewell.codes.icd10 import CodeSet
from notewell.config import ROOT, Settings
from notewell.dataset import Case, load_cases
from notewell.llm.fake import FakeScribe
from notewell.llm.resilient import ResilientModel
from notewell.models import Patient, Turn
from notewell.services import Services, build_services

DATASET = ROOT / "data" / "visits"


@pytest.fixture(scope="session")
def cases() -> list[Case]:
    return load_cases(DATASET)


@pytest.fixture(scope="session")
def case_by_id(cases: list[Case]) -> dict[str, Case]:
    return {c.id: c for c in cases}


@pytest.fixture(scope="session")
def codeset() -> CodeSet:
    """The offline subset shipped with the package (what CI has); tests that need the full list skip without it."""
    return CodeSet.load(None)


@pytest.fixture(scope="session")
def full_codeset() -> CodeSet:
    cs = CodeSet.load(ROOT / "data" / "codes")
    if not cs.complete:
        pytest.skip("the full ICD-10-CM list is not downloaded (notewell codes download)")
    return cs


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "var",
        llm_provider="fake",
        llm_ledger=None,
        seed_demo=True,
        results_dir=tmp_path / "results",
        codes_dir=tmp_path / "codes",
        audio_dir=tmp_path / "audio",
        fhir_base_url="http://127.0.0.1:8000/mock-fhir",
    )


@pytest.fixture
def services(settings: Settings) -> Iterator[Services]:
    yield build_services(settings, llm=ResilientModel(FakeScribe()))


def turns(*pairs: tuple[str, str]) -> list[Turn]:
    return [Turn(idx=i, speaker=s, text=t) for i, (s, t) in enumerate(pairs)]  # type: ignore[arg-type]


PATIENT = Patient(
    name="Marisol Ferreira",
    dob="1971-03-22",
    sex="female",
    mrn="4827163",
    phone="614-555-0187",
    address="1842 Larkspur Lane, Dayton, OH 45419",
    email="marisol.f@example.com",
)
