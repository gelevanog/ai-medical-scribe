from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from notewell.codes.icd10 import CodeSet
from notewell.config import ROOT, Settings
from notewell.dataset import Case
from notewell.eval.asr_eval import align, normalize, score
from notewell.eval.guard_eval import run_guard_eval
from notewell.eval.note_eval import aggregate, match_fact, score_note, statement_polarity
from notewell.eval.phi_eval import evaluate_challenge
from notewell.eval.report import build_report
from notewell.eval.runner import CANARY_VALUES, RunConfig, canary_case, find_leaks, run_canary, run_notes
from notewell.llm.fake import FakeScribe
from notewell.models import Note, Statement, Turn
from notewell.speech.audio import SAMPLE_RATE, add_noise, log_mel, pitch_hz, read_wav, resample, write_wav
from notewell.speech.diarize import Word, chunk_words, diarize
from tests.conftest import turns


def test_wer_normalizer_and_alignment() -> None:
    assert normalize("Two weeks, 152/94!") == ["2", "weeks", "152", "over", "94"]
    assert normalize("Um, mm-hmm, the 3rd") == ["the", "3"]
    a = align(["a", "b", "c"], ["a", "x", "c", "d"])
    assert (a.substitutions, a.deletions, a.insertions) == (1, 0, 1)


def test_asr_score_with_roles() -> None:
    gold = turns(("clinician", "Any chest pain?"), ("patient", "No, I take lisinopril 20 milligrams."))
    hyp = [Turn(idx=0, speaker="clinician", text="Any chest pain? No,"), Turn(idx=1, speaker="patient", text="I take lisinopril 20 milligrams.")]
    s = score(gold, hyp)
    assert s.wer == 0.0
    assert s.key_terms == 2 and s.key_terms_correct == 2  # lisinopril and 20
    assert s.role_correct == s.role_words - 1  # "No" was attributed to the clinician


def test_fact_matching_and_polarity() -> None:
    groups = [["chest pain", "chest pressure"]]
    m = match_fact("Denies chest pain or pressure", groups)
    assert m is not None and statement_polarity("Denies chest pain or pressure", m) == "absent"
    m2 = match_fact("Chest pressure with exertion for 3 days", groups)
    assert m2 is not None and statement_polarity("Chest pressure with exertion for 3 days", m2) == "present"
    assert match_fact("Headache", groups) is None
    assert match_fact("Wheezing at night", [["wheez"]]) is not None


def test_score_note_against_gold(case_by_id: dict[str, Case]) -> None:
    case = case_by_id["pc-01"]
    statements = [Statement(id=f"s{i}", section=f.section, text=("Denies " if f.polarity == "absent" else "") + f.text, support="supported")
                  for i, f in enumerate(case.gold.facts)]  # fmt: skip
    perfect = score_note(case, Note(visit_id=case.id, statements=statements))
    assert perfect.found >= 0.9 * perfect.facts
    flipped = [s.model_copy(update={"text": s.text.replace("Denies ", "Reports ")}) for s in statements]
    bad = score_note(case, Note(visit_id=case.id, statements=flipped))
    assert len(bad.negation_errors) >= 1
    summary = aggregate([perfect, bad])
    assert summary["visits"] == 2 and summary["fact_recall"] is not None


def test_run_notes_and_canary_with_the_fake_model(cases: list[Case], codeset: CodeSet, tmp_path: Path) -> None:
    subset = cases[:4]
    summary = run_notes(subset, FakeScribe(), codeset, RunConfig("fake", tmp_path / "fake"))
    assert summary["visits"] == 4 and summary["phi_leaks"] == 0 and summary["payloads_checked"] == 8
    assert (tmp_path / "fake" / "notes" / f"{subset[0].id}.json").exists()
    canary = run_canary(subset, FakeScribe(), codeset, tmp_path)
    assert canary["leaks"] == 0 and canary["payloads_checked"] == 8


def test_canary_case_injects_and_renames(case_by_id: dict[str, Case]) -> None:
    case = case_by_id["pc-01"]
    c = canary_case(case)
    text = " ".join(t.text for t in c.dialogue)
    assert "Quorinna" in text and "Thaddeus Wrenfield" in text and case.patient.name.split()[0] not in text
    payload = [[{"role": "user", "content": "Thaddeus called from 555 0199"}]]
    assert {x["value"] for x in find_leaks(CANARY_VALUES, payload)} == {"Thaddeus", "555-0199"}


def test_challenge_sets_are_well_formed() -> None:
    for name in ("phi_challenge.yaml", "phi_challenge_2.yaml"):
        items = yaml.safe_load((ROOT / "data" / name).read_text())
        assert len(items) >= 100
        for item in items:
            for phi in item.get("phi") or []:
                assert phi["value"] in item["text"], (item["id"], phi["value"])
        result = evaluate_challenge(items[:10])
        assert result["items"] == 10


def test_guard_eval_and_report(cases: list[Case], codeset: CodeSet, tmp_path: Path) -> None:
    settings = Settings(results_dir=tmp_path, llm_ledger=None, codes_dir=tmp_path / "codes")
    run_notes(cases[:6], FakeScribe(), codeset, RunConfig("main", tmp_path / "runs" / "main"))
    result = run_guard_eval(settings, "main")
    assert result["overall"]["trials"] > 10
    assert result["catch_rate"]["icd_fake"]["rate"] == 1.0
    assert result["catch_rate"]["dose_x10"]["rate"] >= 0.8
    path = build_report(settings)
    assert "Notes run `main`" in path.read_text()
    assert json.loads((tmp_path / "summary.json").read_text())["runs"]["main"]["visits"] == 6


def test_wav_roundtrip_resample_and_noise(tmp_path: Path) -> None:
    t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    write_wav(tmp_path / "a.wav", tone)
    back, rate = read_wav(tmp_path / "a.wav")
    assert rate == SAMPLE_RATE and np.allclose(back, tone, atol=1e-3)
    assert len(resample(tone, SAMPLE_RATE, 8000)) == 8000
    noisy = add_noise(tone, 10.0, np.random.default_rng(0))
    snr = 10 * np.log10(np.mean(tone**2) / np.mean((noisy - tone) ** 2))
    assert 8.5 < snr < 11.5
    assert abs(pitch_hz(tone) - 220) < 10
    assert log_mel(tone).shape[1] == 40


def _voice(f0: float, seconds: float) -> np.ndarray:  # type: ignore[type-arg]
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    harmonics = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    return (0.2 * harmonics * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def test_diarization_separates_two_voices_and_assigns_roles() -> None:
    lines = [
        (110.0, "Good morning. Any chest pain or shortness of breath?"),
        (230.0, "No, not really. I've been tired though."),
        (110.0, "Okay. Let's check your blood pressure. Take a deep breath."),
        (230.0, "My back hurts when I sit too long, and I feel worn out."),
        (110.0, "I'd like to start a new medicine, 10 milligrams daily. Any questions?"),
        (230.0, "Yeah, um, is it safe with my other pills?"),
    ]
    audio_parts, words, cursor = [], [], 0.0
    for f0, text in lines:
        seg = _voice(f0, 2.4)
        tokens = text.split()
        step = 2.4 / len(tokens)
        for i, tok in enumerate(tokens):
            words.append(Word(text=(" " if i else " ") + tok, start=cursor + i * step, end=cursor + (i + 0.8) * step))
        audio_parts += [seg, np.zeros(int(0.6 * SAMPLE_RATE), dtype=np.float32)]
        cursor += 3.0
    audio = np.concatenate(audio_parts)
    assert len(chunk_words(words)) >= 6
    result, info = diarize(audio, words)
    assert [t.speaker for t in result] == ["clinician", "patient"] * 3
    assert info["chunks"] >= 6


@pytest.mark.speech
def test_whisper_transcribes_a_synthetic_visit(settings: Settings) -> None:
    pytest.importorskip("faster_whisper")
    path = ROOT / "data" / "audio" / "pc-01.wav"
    if not path.exists():
        pytest.skip("synthetic audio not generated (notewell data synth-audio)")
    from notewell.speech.asr import WhisperASR

    audio, _ = read_wav(path)
    transcript = WhisperASR("tiny.en", threads=4).transcribe(audio[: SAMPLE_RATE * 30])
    assert transcript.turns and transcript.timings["asr"] > 0
