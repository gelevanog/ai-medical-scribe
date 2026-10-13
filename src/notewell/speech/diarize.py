"""Two-speaker diarization and role inference without a diarization model.

1. **Chunks**: Whisper's words are cut where the speech pauses (>= 0.3 s) or a sentence ends before a short pause.
2. **Voices**: each chunk gets an acoustic signature: mean and spread of 40 log-mel bands plus the median pitch.
   Signatures are standardized and split into two clusters with k-means (several restarts); a chunk too short to
   carry a reliable signature inherits the cluster of its neighbours.
3. **Roles**: the cluster that asks more questions and uses more clinician phrasing ("any ... ?", "I'd like to",
   "let's", "your blood pressure", "milligrams") is the clinician; the other one is the patient (or the caregiver).

It is a deliberately small method for a two-person exam room. It is measured on the synthetic audio below
(word-level role accuracy); a production system with more speakers needs a real diarization model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from notewell.models import Role, Turn
from notewell.speech.audio import SAMPLE_RATE, Audio, log_mel, pitch_hz


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0


@dataclass
class Chunk:
    words: list[Word]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        # Whisper's word strings carry their own leading space ("follow" + "-up" stays "follow-up").
        return "".join(
            w.text if w.text[:1].isspace() or i == 0 else " " + w.text for i, w in enumerate(self.words)
        ).strip()


_CLINICIAN = re.compile(
    r"\b(?:any|do you|have you|are you|did you|how (?:long|much|often|are|is)|what (?:brings|happened)|when did|"
    r"let's|let me|i'd like|i want you|i'm going to|we'll|we will|i recommend|i think|your (?:blood pressure|"
    r"weight|pulse|heart|lungs|exam|labs?|a1c|temperature|score)|milligrams?|prescri\w+|follow up|come back|"
    r"take a (?:look|listen)|deep breath|on exam|looks|sounds|go ahead|confirm your|date of birth)\b",
    re.I,
)
_PATIENT = re.compile(
    r"\b(?:i've been|i have been|i've had|my (?:back|head|chest|stomach|knee|skin|husband|wife|son|daughter|mom|"
    r"dad|job|work)|it hurts|i feel|i don't know|i guess|i think so|i noticed|i was|it started|it's been|"
    r"yeah|no,|not really|um|uh|kind of|sort of|she's been|he's been|she has|he has)\b",
    re.I,
)
MIN_CHUNK_S = 0.8


def chunk_words(words: list[Word], *, pause: float = 0.3, sentence_pause: float = 0.12) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[Word] = []
    for w in words:
        if current:
            gap = w.start - current[-1].end
            ends_sentence = current[-1].text.rstrip().endswith((".", "?", "!"))
            if gap >= pause or (ends_sentence and gap >= sentence_pause):
                chunks.append(Chunk(current))
                current = []
        current.append(w)
    if current:
        chunks.append(Chunk(current))
    return chunks


def signature(audio: Audio, start: float, end: float) -> np.ndarray[Any, np.dtype[np.float64]]:
    a = audio[max(0, int(start * SAMPLE_RATE)) : int(end * SAMPLE_RATE)]
    mel = log_mel(a)
    if len(mel) < 5:
        return np.zeros(81)
    energy = mel.mean(axis=1)
    voiced = mel[energy > np.percentile(energy, 40)] if len(mel) > 10 else mel
    f0 = pitch_hz(a)
    return np.concatenate([voiced.mean(axis=0), voiced.std(axis=0), [np.log(f0 + 1.0) * 4.0]])


def kmeans2(
    x: np.ndarray[Any, np.dtype[np.float64]], *, restarts: int = 12, seed: int = 0
) -> np.ndarray[Any, np.dtype[np.int64]]:
    rng = np.random.default_rng(seed)
    best_labels = np.zeros(len(x), dtype=np.int64)
    best_inertia = np.inf
    for _ in range(restarts):
        centers = x[rng.choice(len(x), 2, replace=False)]
        labels = np.zeros(len(x), dtype=np.int64)
        for _ in range(50):
            dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new = dist.argmin(axis=1)
            if np.array_equal(new, labels) and _ > 0:
                break
            labels = new
            for k in range(2):
                if np.any(labels == k):
                    centers[k] = x[labels == k].mean(axis=0)
        inertia = float(((x - centers[labels]) ** 2).sum())
        if inertia < best_inertia and len(set(labels.tolist())) == 2:
            best_inertia, best_labels = inertia, labels.copy()
    return best_labels


def role_scores(chunks: list[Chunk], labels: np.ndarray[Any, np.dtype[np.int64]]) -> dict[int, float]:
    scores: dict[int, float] = {0: 0.0, 1: 0.0}
    for chunk, label in zip(chunks, labels, strict=True):
        text = chunk.text
        clinician = len(_CLINICIAN.findall(text)) + 1.5 * text.count("?")
        patient = len(_PATIENT.findall(text))
        scores[int(label)] += clinician - patient
    return scores


def diarize(audio: Audio, words: list[Word], *, second_speaker: str = "patient") -> tuple[list[Turn], dict[str, Any]]:
    chunks = chunk_words(words)
    if not chunks:
        return [], {"chunks": 0}
    other: Role = "caregiver" if second_speaker == "caregiver" else "patient"
    if len(chunks) < 3:
        labels = np.zeros(len(chunks), dtype=np.int64)
    else:
        feats = np.stack([signature(audio, c.start, c.end) for c in chunks])
        long_enough = np.array([(c.end - c.start) >= MIN_CHUNK_S for c in chunks])
        fit = feats[long_enough] if long_enough.sum() >= 4 else feats
        mean, std = fit.mean(axis=0), fit.std(axis=0) + 1e-6
        z = (feats - mean) / std
        labels = kmeans2(z[long_enough] if long_enough.sum() >= 4 else z)
        if long_enough.sum() >= 4:
            centers = np.stack([z[long_enough][labels == k].mean(axis=0) for k in range(2)])
            full = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
            # Short chunks ("Yes.", "Okay.") carry little voice: keep the nearest center only if it is clear,
            # otherwise the speaker alternates (a short reply follows the other speaker).
            labels = full
            for i, ok in enumerate(long_enough):
                if not ok and 0 < i < len(chunks) - 1 and labels[i - 1] == labels[i + 1]:
                    dist = ((z[i] - centers) ** 2).sum(axis=1)
                    if abs(dist[0] - dist[1]) < 0.15 * dist.sum():
                        labels[i] = 1 - labels[i - 1]
    scores = role_scores(chunks, labels)
    clinician_label = max(scores, key=lambda k: scores[k])
    turns: list[Turn] = []
    for chunk, label in zip(chunks, labels, strict=True):
        role: Role = "clinician" if int(label) == clinician_label else other
        if turns and turns[-1].speaker == role:
            last = turns[-1]
            turns[-1] = last.model_copy(update={"text": f"{last.text} {chunk.text}", "end": round(chunk.end, 2)})
        else:
            turns.append(
                Turn(
                    idx=len(turns), speaker=role, text=chunk.text, start=round(chunk.start, 2), end=round(chunk.end, 2)
                )
            )
    info = {
        "method": "log-mel + pitch signatures, k-means (k=2), lexical role inference",
        "chunks": len(chunks),
        "role_scores": {"clinician": round(scores[clinician_label], 1), other: round(scores[1 - clinician_label], 1)},
    }
    return turns, info
