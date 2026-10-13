"""Local speech on CPU: faster-whisper transcription, a small acoustic diarizer, Kokoro TTS for the evaluation audio.

The runtimes are the optional `speech` extra (`uv sync --extra speech`) and the weights are downloaded separately
(`notewell models download`); nothing here is needed for the offline demo, which uses the gold transcripts.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    license: str
    used_for: str


KOKORO_FILES = [
    ModelFile(
        "kokoro-v1.0.onnx",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "Apache-2.0 (Kokoro-82M weights, hexgrad)",
        "synthetic visit audio (TTS), evaluation only",
    ),
    ModelFile(
        "voices-v1.0.bin",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        "Apache-2.0 (Kokoro-82M voices)",
        "synthetic visit audio (TTS), evaluation only",
    ),
]


def download_models(models_dir: Path, *, whisper: list[str], kokoro: bool = True, log: object = print) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    printer = log if callable(log) else print
    if kokoro:
        for item in KOKORO_FILES:
            target = models_dir / item.name
            if target.exists() and target.stat().st_size > 0:
                printer(f"ok        {item.name} ({item.license})")
                continue
            printer(f"download  {item.name} <- {item.url}")
            partial = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(item.url, partial)  # noqa: S310 - fixed https URL
            partial.rename(target)
    for name in whisper:
        from faster_whisper import download_model

        printer(f"whisper   {name} -> {download_model(name)}")
