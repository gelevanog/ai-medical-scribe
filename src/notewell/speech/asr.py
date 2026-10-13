"""Speech to text with faster-whisper on CPU (int8), word timestamps, then diarization into clinician/patient turns."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from notewell.models import Transcript
from notewell.speech.audio import SAMPLE_RATE, Audio
from notewell.speech.diarize import Word, diarize

# A fixed list of commonly prescribed US drugs and clinical words (not taken from the evaluation set) that primes
# Whisper's spelling of medication names.
MEDICAL_PROMPT = (
    "Clinic visit. Medications: lisinopril, amlodipine, metformin, atorvastatin, rosuvastatin, levothyroxine, "
    "metoprolol, losartan, hydrochlorothiazide, furosemide, apixaban, sertraline, escitalopram, bupropion, "
    "gabapentin, omeprazole, albuterol, fluticasone, montelukast, amoxicillin, cephalexin, azithromycin, "
    "doxycycline, prednisone, ibuprofen, acetaminophen. Blood pressure 128 over 82, A1c, PHQ-9."
)


@dataclass
class ASRResult:
    words: list[Word]
    language: str
    duration_s: float
    asr_s: float


class WhisperASR:
    def __init__(self, model: str = "small.en", *, threads: int = 4, medical_prompt: bool = True) -> None:
        from faster_whisper import WhisperModel

        self.model_name = model
        self.medical_prompt = medical_prompt
        self.model = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)

    def words(self, audio: Audio) -> ASRResult:
        started = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            language="en",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
            initial_prompt=MEDICAL_PROMPT if self.medical_prompt else None,
        )
        words: list[Word] = []
        for segment in segments:
            for w in segment.words or []:
                if w.word.strip():
                    words.append(Word(text=w.word, start=float(w.start), end=float(w.end), prob=float(w.probability)))
        return ASRResult(words, info.language, len(audio) / SAMPLE_RATE, time.perf_counter() - started)

    def transcribe(self, audio: Audio, *, second_speaker: str = "patient") -> Transcript:
        result = self.words(audio)
        t0 = time.perf_counter()
        turns, info = diarize(audio, result.words, second_speaker=second_speaker)
        return Transcript(
            turns=turns,
            source="asr",
            asr_model=f"faster-whisper {self.model_name} int8{' + medical prompt' if self.medical_prompt else ''}",
            duration_s=round(result.duration_s, 2),
            timings={"asr": round(result.asr_s, 2), "diarization": round(time.perf_counter() - t0, 2)},
            diarization=info,
        )


def decode_audio(path: Path) -> Audio:
    """Any container/codec (WAV, WebM/Opus from the browser recorder, MP3) to 16 kHz mono float32, via PyAV."""
    from faster_whisper import decode_audio as fw_decode

    audio: Any = fw_decode(str(path), sampling_rate=SAMPLE_RATE)
    return np.asarray(audio, dtype=np.float32)
