#!/usr/bin/env python3
"""Generate EQ-style visualizations from an audio file (librosa + matplotlib)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "eq_outputs"

LOG_FREQ_TICKS = (100, 1000, 10000)
LOG_FREQ_LABELS = ("100", "1000", "10000")


def format_log_frequency_axis(ax: plt.Axes, axis: str = "x") -> None:
    """Use plain Hz labels (100, 1000, 10000) instead of 10²-style notation."""
    if axis == "x":
        ax.set_xticks(LOG_FREQ_TICKS)
        ax.set_xticklabels(LOG_FREQ_LABELS)
    else:
        ax.set_yticks(LOG_FREQ_TICKS)
        ax.set_yticklabels(LOG_FREQ_LABELS)


def load_audio(path: Path, sr: int | None) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise SystemExit(f"Audio file not found: {path}")
    y, loaded_sr = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        raise SystemExit(f"Audio file is empty: {path}")
    return y, loaded_sr


def plot_spectrogram(
    y: np.ndarray,
    sr: int,
    ax: plt.Axes,
    *,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> None:
    _ = n_mels  # CLI flag kept; log-freq STFT used for Hz axis labels
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
    n_fft: int,
    hop_length: int,
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


def build_figure(
    y: np.ndarray,
    sr: int,
    plot_mode: str,
    *,
    n_fft: int,
    hop_length: int,
    n_mels: int,
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


def default_output_path(audio_path: Path, plot_mode: str) -> Path:
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUT_DIR / f"{audio_path.stem}_{plot_mode}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate spectrogram / average spectrum EQ plots from an audio file.",
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to input audio (wav, mp3, flac, etc.).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: text_pair_gen/eq_outputs/<name>_<mode>.png",
    )
    parser.add_argument(
        "--plot",
        choices=("both", "spectrogram", "spectrum"),
        default="both",
        help="both = mel spectrogram + average spectrum (default).",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Target sample rate (default: keep native rate).",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=4096,
        help="FFT size for STFT / spectrum.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=1024,
        help="STFT hop length in samples.",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Number of mel bands for spectrogram.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive plot window after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = args.audio.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else default_output_path(audio_path, args.plot)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y, sr = load_audio(audio_path, args.sr)
    duration = librosa.get_duration(y=y, sr=sr)

    print(f"Input:  {audio_path}")
    print(f"Duration: {duration:.2f}s @ {sr} Hz")
    print(f"Plot:   {args.plot}")

    fig = build_figure(
        y,
        sr,
        args.plot,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved:  {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
