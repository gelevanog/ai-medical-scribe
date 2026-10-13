"""PHI detection numbers for the report: the dialogues (gold and ASR transcripts) and the held-out challenge sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from notewell.config import Settings
from notewell.dataset import load_cases
from notewell.eval.phi_eval import evaluate, evaluate_challenge
from notewell.models import Transcript, Turn


def run_phi(settings: Settings) -> dict[str, Any]:
    out = settings.results_dir / "phi"
    out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(settings.dataset_dir)
    summary: dict[str, Any] = {}
    dialogues = evaluate(cases).to_dict()
    summary["dialogues_gold_text"] = {k: v for k, v in dialogues.items() if k != "over_masked_examples"}
    asr_dir = settings.results_dir / "asr" / "transcripts" / "small.en-clean"
    if asr_dir.exists():
        turns: dict[str, list[Turn]] = {}
        for case in cases:
            path = asr_dir / f"{case.id}.json"
            if path.exists():
                turns[case.id] = Transcript.model_validate_json(path.read_text(encoding="utf-8")).turns
        with_asr = [c for c in cases if c.id in turns]
        result = evaluate(with_asr, turns_by_case=turns).to_dict()
        summary["dialogues_asr_text"] = {k: v for k, v in result.items() if k != "over_masked_examples"}
    for name in ("phi_challenge.yaml", "phi_challenge_2.yaml"):
        path = settings.dataset_dir.parent / name
        if path.exists():
            items = yaml.safe_load(path.read_text(encoding="utf-8"))
            r = evaluate_challenge(items)
            summary[Path(name).stem + "_current"] = {
                k: v for k, v in r.items() if k not in {"misses", "over_masked_negatives"}
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return {
        k: {kk: vv for kk, vv in v.items() if kk in {"recall", "gold_values", "covered"}} for k, v in summary.items()
    }
