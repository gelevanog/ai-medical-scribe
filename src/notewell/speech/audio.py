"""Small audio helpers with numpy only: WAV I/O (16-bit PCM), resampling, noise at a target SNR, log-mel features."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Audio = NDArray[np.float32]
SAMPLE_RATE = 16000


def write_wav(path: Path, audio: Audio, rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def read_wav(data: bytes | Path) -> tuple[Audio, int]:
    raw = data.read_bytes() if isinstance(data, Path) else data
    with wave.open(io.BytesIO(raw), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError("only 16-bit PCM WAV is supported here")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32), rate


def resample(audio: Audio, rate: int, target: int = SAMPLE_RATE) -> Audio:
    if rate == target:
        return audio.astype(np.float32)
    duration = len(audio) / rate
    n = round(duration * target)
    # Windowed-sinc would be better; linear interpolation after a crude low-pass is enough for 24k -> 16k speech.
    if target < rate:
        kernel = np.hanning(int(rate / target) * 4 + 1)
        kernel /= kernel.sum()
        audio = np.convolve(audio, kernel, mode="same").astype(np.float32)
    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> Audio:
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.arange(len(spectrum))
    spectrum[1:] /= np.sqrt(freqs[1:])
    noise = np.fft.irfft(spectrum, n)
    out: Audio = (noise / (np.std(noise) + 1e-9)).astype(np.float32)
    return out


def add_noise(audio: Audio, snr_db: float, rng: np.random.Generator) -> Audio:
    speech_power = float(np.mean(audio[np.abs(audio) > 0.01] ** 2)) if np.any(np.abs(audio) > 0.01) else 1e-4
    noise = pink_noise(len(audio), rng)
    noise *= np.sqrt(speech_power / (10 ** (snr_db / 10)))
    noisy: Audio = np.clip(audio + noise, -1.0, 1.0).astype(np.float32)
    return noisy


def _mel_filters(n_fft: int, n_mels: int, rate: int) -> NDArray[np.float64]:
    def hz_to_mel(f: NDArray[np.float64]) -> NDArray[np.float64]:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: NDArray[np.float64]) -> NDArray[np.float64]:
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(np.array(60.0)), hz_to_mel(np.array(rate / 2 - 200)), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / rate).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / max(1, right - center)
    return fb


_FILTERS: dict[tuple[int, int, int], NDArray[np.float64]] = {}


def log_mel(audio: Audio, rate: int = SAMPLE_RATE, n_mels: int = 40) -> NDArray[np.float64]:
    """Frames x mels log-mel energies (25 ms windows, 10 ms hop)."""
    win, hop, n_fft = int(0.025 * rate), int(0.010 * rate), 512
    if len(audio) < win:
        return np.zeros((0, n_mels))
    key = (n_fft, n_mels, rate)
    if key not in _FILTERS:
        _FILTERS[key] = _mel_filters(n_fft, n_mels, rate)
    frames = np.lib.stride_tricks.sliding_window_view(audio, win)[::hop] * np.hanning(win)
    power = np.abs(np.fft.rfft(frames, n_fft)) ** 2
    return np.log(power @ _FILTERS[key].T + 1e-8)


def pitch_hz(audio: Audio, rate: int = SAMPLE_RATE) -> float:
    """Median fundamental frequency of voiced 40 ms frames (autocorrelation), 0 if nothing is voiced."""
    win, hop = int(0.04 * rate), int(0.02 * rate)
    lo, hi = int(rate / 400), int(rate / 70)
    f0s: list[float] = []
    for start in range(0, max(0, len(audio) - win), hop):
        frame = audio[start : start + win]
        if float(np.sqrt(np.mean(frame**2))) < 0.02:
            continue
        frame = frame - frame.mean()
        corr = np.correlate(frame, frame, mode="full")[win - 1 :]
        if corr[0] <= 0:
            continue
        lag = lo + int(np.argmax(corr[lo:hi]))
        if corr[lag] / corr[0] > 0.45:
            f0s.append(rate / lag)
    return float(np.median(f0s)) if f0s else 0.0
