"""Synthesize the evaluation audio: each visit dialogue spoken by two Kokoro voices (CPU), with natural gaps between
turns, saved clean and with pink noise at 15 dB SNR. The gold turn timings go into a JSON file next to the audio."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from notewell.dataset import Case
from notewell.speech.audio import SAMPLE_RATE, add_noise, resample, write_wav

VOICES = {
    "female": ["af_heart", "af_bella", "bf_emma", "af_nicole", "af_sarah", "bf_isabella"],
    "male": ["am_michael", "am_fenrir", "bm_george", "am_puck", "am_echo", "bm_lewis"],
}
NOISE_SNR_DB = 15.0


def pick_voices(case: Case) -> dict[str, str]:
    """Deterministic, different voices for the two speakers of a visit."""
    seed = int(hashlib.sha256(case.id.encode()).hexdigest(), 16)
    chosen: dict[str, str] = {}
    for i, (speaker, profile) in enumerate(sorted(case.speakers.items())):
        pool = VOICES[profile.get("sex", "female")]
        voice = pool[(seed // (7**i)) % len(pool)]
        while voice in chosen.values():
            voice = pool[(pool.index(voice) + 1) % len(pool)]
        chosen[speaker] = voice
    return chosen


def speakable(text: str) -> str:
    """Tweak written forms that Kokoro reads badly; the gold transcript keeps the original text."""
    out = text.replace("A1c", "A 1 C").replace("PHQ-9", "P H Q 9").replace("GAD-7", "G A D 7")
    out = re.sub(r"\b(\d{3})-(\d{4})\b", lambda m: " ".join(m.group(1)) + ", " + " ".join(m.group(2)), out)
    return out


class Synthesizer:
    def __init__(self, models_dir: Path, *, speed: float = 1.0) -> None:
        from kokoro_onnx import Kokoro

        self.engine = Kokoro(str(models_dir / "kokoro-v1.0.onnx"), str(models_dir / "voices-v1.0.bin"))
        self.speed = speed

    def turn_audio(self, text: str, voice: str) -> np.ndarray[Any, np.dtype[np.float32]]:
        samples, rate = self.engine.create(speakable(text), voice=voice, speed=self.speed, lang="en-us")
        return resample(np.asarray(samples, dtype=np.float32), rate, SAMPLE_RATE)

    def synthesize(self, case: Case, out_dir: Path) -> dict[str, Any]:
        rng = np.random.default_rng(int(hashlib.sha256(case.id.encode()).hexdigest()[:8], 16))
        voices = pick_voices(case)
        pieces: list[np.ndarray[Any, np.dtype[np.float32]]] = [np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32)]
        cursor = 0.4
        timings: list[dict[str, Any]] = []
        started = time.perf_counter()
        for turn in case.dialogue:
            audio = self.turn_audio(turn.text, voices[turn.speaker])
            peak = float(np.max(np.abs(audio))) or 1.0
            audio = (audio / peak * 0.8).astype(np.float32)
            duration = len(audio) / SAMPLE_RATE
            timings.append(
                {
                    "idx": turn.idx,
                    "speaker": turn.speaker,
                    "start": round(cursor, 3),
                    "end": round(cursor + duration, 3),
                }
            )
            pieces.append(audio)
            gap = float(rng.uniform(0.25, 0.75))
            pieces.append(np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32))
            cursor += duration + gap
        clean = np.concatenate(pieces)
        write_wav(out_dir / f"{case.id}.wav", clean)
        write_wav(out_dir / f"{case.id}.noisy.wav", add_noise(clean, NOISE_SNR_DB, rng))
        meta = {
            "id": case.id,
            "voices": voices,
            "duration_s": round(len(clean) / SAMPLE_RATE, 2),
            "synthesis_s": round(time.perf_counter() - started, 1),
            "sample_rate": SAMPLE_RATE,
            "noise": {"type": "pink", "snr_db": NOISE_SNR_DB},
            "turns": timings,
        }
        (out_dir / f"{case.id}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return meta
