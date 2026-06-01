"""EQ plots and audio metrics for dataset generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

LOG_FREQ_TICKS = (100, 1000, 10000)
LOG_FREQ_LABELS = ("100", "1000", "10000")

# Metric keys used in aug_{i}_metrics.json
METRIC_KEYS = (
    "dry_vocal",
    "dry_instrumental",
    "wet_vocal",
    "wet_instrumental",
    "src_sep_vocal",
    "src_sep_instrumental",
    "mix",
)


def format_log_frequency_axis(ax: plt.Axes, axis: str = "x") -> None:
    if axis == "x":
        ax.set_xticks(LOG_FREQ_TICKS)
        ax.set_xticklabels(LOG_FREQ_LABELS)
    else:
        ax.set_yticks(LOG_FREQ_TICKS)
        ax.set_yticklabels(LOG_FREQ_LABELS)


def load_wav_mono(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load audio and return mono (time,) for analysis."""
    data, loaded_sr = sf.read(str(path), always_2d=True)
    if sr is not None and loaded_sr != sr:
        data = librosa.resample(data.T, orig_sr=loaded_sr, target_sr=sr).T
        loaded_sr = sr
    if data.shape[1] == 1:
        y = data[:, 0]
    else:
        y = data.mean(axis=1)
    if y.size == 0:
        raise ValueError(f"Empty audio: {path}")
    return y.astype(np.float64), loaded_sr


def wav_to_stereo_np(path: Path) -> np.ndarray:
    """Return (channels, time) stereo numpy from wav."""
    data, _ = sf.read(str(path), always_2d=True)
    if data.shape[1] == 1:
        return np.stack([data[:, 0], data[:, 0]], axis=0)
    return data.T


def get_crest_factor_db(waveform: np.ndarray) -> float:
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    peak = np.max(np.abs(waveform), axis=-1)
    rms = np.sqrt(np.mean(waveform**2, axis=-1))
    cf_db = 20 * np.log10(peak / (rms + 1e-8))
    return float(np.mean(cf_db))


def get_panning_norm(waveform: np.ndarray) -> float:
    if waveform.ndim != 2 or waveform.shape[0] != 2:
        raise ValueError("Waveform must be stereo (2, time)")
    left, right = waveform[0], waveform[1]
    left_energy = np.sum(left**2)
    right_energy = np.sum(right**2)
    return float((right_energy - left_energy) / (right_energy + left_energy + 1e-8))


def get_lufs(waveform: np.ndarray, sr: int) -> float:
    meter = pyln.Meter(sr)
    if waveform.ndim == 1:
        data = waveform[:, np.newaxis]
    elif waveform.shape[0] < waveform.shape[1]:
        data = waveform.transpose()
    else:
        data = waveform
    try:
        return float(meter.integrated_loudness(data))
    except ValueError:
        return float("-inf")


def get_transient_stats(y_mono: np.ndarray, sr: int) -> dict[str, float]:
    onset = librosa.onset.onset_strength(y=y_mono, sr=sr)
    return {
        "mean": float(np.mean(onset)),
        "max": float(np.max(onset)),
        "std": float(np.std(onset)),
    }


def compute_metrics_for_wav(path: Path, sr: int) -> dict[str, Any]:
    stereo = wav_to_stereo_np(path)
    y_mono, _ = load_wav_mono(path, sr=sr)
    pan_norm = get_panning_norm(stereo)
    return {
        "lufs_db": get_lufs(stereo, sr),
        "panning": round(pan_norm * 100.0, 2),
        "crest_factor_db": round(get_crest_factor_db(stereo), 2),
        "transient": get_transient_stats(y_mono, sr),
        "path": str(path),
    }


def plot_spectrogram(
    y: np.ndarray,
    sr: int,
    ax: plt.Axes,
    *,
    n_fft: int = 4096,
    hop_length: int = 1024,
    n_mels: int = 128,
) -> None:
    _ = n_mels
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    img = librosa.display.specshow(
        stft_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="log",
        y_axis="time",
        ax=ax,
    )
    format_log_frequency_axis(ax, axis="x")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title("Log-frequency spectrogram")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")


def plot_average_spectrum(
    y: np.ndarray,
    sr: int,
    ax: plt.Axes,
    *,
    n_fft: int = 4096,
    hop_length: int = 1024,
) -> None:
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    avg_mag = magnitude.mean(axis=1)
    avg_db = librosa.amplitude_to_db(avg_mag, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    ax.semilogx(freqs[1:], avg_db[1:])
    ax.set_xlim(20, sr / 2)
    format_log_frequency_axis(ax, axis="x")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Average magnitude spectrum (EQ-style curve)")
    ax.grid(True, which="both", alpha=0.3)


def build_eq_figure(
    y: np.ndarray,
    sr: int,
    plot_mode: str = "both",
    *,
    n_fft: int = 4096,
    hop_length: int = 1024,
    n_mels: int = 128,
) -> plt.Figure:
    if plot_mode == "spectrogram":
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_spectrogram(y, sr, ax, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    elif plot_mode == "spectrum":
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_average_spectrum(y, sr, ax, n_fft=n_fft, hop_length=hop_length)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        plot_spectrogram(y, sr, axes[0], n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        plot_average_spectrum(y, sr, axes[1], n_fft=n_fft, hop_length=hop_length)
    fig.suptitle(f"Audio analysis (sr={sr} Hz)", fontsize=11)
    fig.tight_layout()
    return fig


def save_eq_plot(
    audio_path: Path,
    output_path: Path,
    sr: int | None = None,
    plot_mode: str = "both",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y, loaded_sr = load_wav_mono(audio_path, sr=sr)
    fig = build_eq_figure(y, loaded_sr, plot_mode=plot_mode)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_aug_analysis(
    song_dir: Path,
    aug_idx: int,
    wav_paths: dict[str, Path],
    sr: int,
    force: bool = False,
) -> dict[str, Any]:
    """
    Generate EQ PNGs under song_dir/eq/ and aug_{i}_metrics.json at song root.

    wav_paths: metric_key -> path to wav
    """
    song_dir = Path(song_dir)
    eq_dir = song_dir / "eq"
    metrics_path = song_dir / f"aug_{aug_idx}_metrics.json"

    if metrics_path.is_file() and not force:
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)

    eq_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {"aug_idx": aug_idx}

    for key, wav_path in wav_paths.items():
        wav_path = Path(wav_path)
        eq_png = eq_dir / f"aug_{aug_idx}_{key}.png"
        save_eq_plot(wav_path, eq_png, sr=sr, plot_mode="both")
        metrics[key] = compute_metrics_for_wav(wav_path, sr)
        metrics[key]["eq_plot"] = str(eq_png)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
