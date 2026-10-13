"""Run faster-whisper + diarization over the synthesized visit audio and score it against the gold dialogues."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from notewell.config import Settings
from notewell.dataset import load_cases
from notewell.eval.asr_eval import ASRScore, score
from notewell.speech.audio import read_wav


def run_asr(
    settings: Settings,
    *,
    model: str = "small.en",
    condition: str = "clean",
    prompt: bool = True,
    ids: list[str] | None = None,
    threads: int = 8,
) -> dict[str, Any]:
    from notewell.speech.asr import WhisperASR

    name = f"{model}-{condition}{'' if prompt else '-noprompt'}"
    out_dir = settings.results_dir / "asr"
    transcripts_dir = out_dir / "transcripts" / name
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    asr = WhisperASR(model, threads=threads, medical_prompt=prompt)
    total = ASRScore()
    per_case: list[dict[str, Any]] = []
    rtf: list[float] = []
    for case in load_cases(settings.dataset_dir, ids=ids):
        suffix = ".noisy.wav" if condition == "noisy" else ".wav"
        path = settings.audio_dir / f"{case.id}{suffix}"
        if not path.exists():
            continue
        audio, _ = read_wav(path)
        second = "caregiver" if "caregiver" in case.speakers else "patient"
        transcript = asr.transcribe(audio, second_speaker=second)
        (transcripts_dir / f"{case.id}.json").write_text(transcript.model_dump_json(indent=1), encoding="utf-8")
        s = score(case.dialogue, transcript.turns)
        total.add(s)
        rtf.append(transcript.timings["asr"] / max(0.1, transcript.duration_s))
        per_case.append(
            {
                "case": case.id,
                "specialty": case.specialty,
                **s.to_dict(),
                "audio_s": transcript.duration_s,
                "asr_s": transcript.timings["asr"],
                "diarization_s": transcript.timings["diarization"],
                "turns_gold": len(case.dialogue),
                "turns_asr": len(transcript.turns),
                "misrecognized_terms": s.misrecognized_terms,
            }
        )
    terms: dict[str, int] = {}
    for c in per_case:
        for k, v in c["misrecognized_terms"].items():
            terms[k] = terms.get(k, 0) + v
    summary = {
        "name": name,
        "model": f"faster-whisper {model} int8 (CPU, {threads} threads)",
        "condition": condition,
        "medical_prompt": prompt,
        "visits": len(per_case),
        **total.to_dict(),
        "audio_minutes": round(sum(c["audio_s"] for c in per_case) / 60, 1),
        "real_time_factor_p50": round(statistics.median(rtf), 3) if rtf else None,
        "asr_seconds_per_visit_p50": round(statistics.median(c["asr_s"] for c in per_case), 1) if per_case else None,
        "diarization_seconds_per_visit_p50": round(statistics.median(c["diarization_s"] for c in per_case), 2)
        if per_case
        else None,
        "misrecognized_drug_terms": dict(sorted(terms.items(), key=lambda kv: -kv[1])[:30]),
        "cases": per_case,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def transcripts_dir(settings: Settings, name: str) -> Path:
    return settings.results_dir / "asr" / "transcripts" / name
