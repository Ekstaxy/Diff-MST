#!/usr/bin/env python3
"""One-shot Gemini test: fixed mix-engineer prompt + optional text / image / audio."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEY_FILE = REPO_ROOT / "gemini_key.txt"

SYSTEM_INSTRUCTION = """\
You are an expert mix engineer. Your task is to translate raw mixing parameters
into natural language descriptions that a musician or engineer would use.

Follow the style of these examples:
- Input: {"track": "vocal", "gain_db": 4.0, "eq": {"high_shelf": 2.5}}
  Output: A bright vocal track with a slight boost in presence and increased overall volume.
- Input: {"track": "drums", "compressor": {"ratio": 8, "threshold": -20}, "panning": 0}
  Output: A heavily compressed, punchy drum bus placed dead center in the mix.

When an EQ curve image is provided, relate the visual shape to tonal changes.
When an audio clip is provided, relate the heard sound to the parameter changes.
Be concise, specific, and use professional mixing language.\
"""

DEFAULT_PARAM_TEXT = json.dumps(
    {
        "track": "vocal",
        "gain_db": 3.0,
        "eq": {"high_shelf_db": 2.0, "freq_hz": 8000},
        "panning": 0.8,
        "compressor": {"ratio": 3, "threshold_db": -18, "attack_ms": 10},
    },
    indent=2,
)

MIME_OVERRIDES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


def load_api_key(explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()
    if DEFAULT_KEY_FILE.is_file():
        return DEFAULT_KEY_FILE.read_text(encoding="utf-8").strip()
    raise SystemExit(
        "No API key: set GEMINI_API_KEY, pass --api-key, or create gemini_key.txt at repo root."
    )


def guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in MIME_OVERRIDES:
        return MIME_OVERRIDES[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    raise SystemExit(f"Could not infer MIME type for {path}. Use a common image/audio extension.")


def load_text_data(args: argparse.Namespace) -> str:
    if args.text_data is not None:
        return args.text_data
    if args.params_file is not None:
        path = Path(args.params_file)
        if not path.is_file():
            raise SystemExit(f"Params file not found: {path}")
        return path.read_text(encoding="utf-8")
    return DEFAULT_PARAM_TEXT


def build_user_parts(
    param_text: str,
    image_path: Path | None,
    audio_path: Path | None,
) -> list[types.Part]:
    parts: list[types.Part] = [
        types.Part.from_text(
            text=(
                "Generate a natural-language mix description for the input below.\n\n"
                "--- Mixing parameters (text) ---\n"
                f"{param_text.strip()}\n"
            )
        ),
    ]

    if image_path is not None:
        if not image_path.is_file():
            raise SystemExit(f"Image not found: {image_path}")
        mime = guess_mime(image_path)
        parts.append(types.Part.from_text(text="--- EQ / spectrum image ---"))
        parts.append(
            types.Part.from_bytes(
                data=image_path.read_bytes(),
                mime_type=mime,
            )
        )

    if audio_path is not None:
        if not audio_path.is_file():
            raise SystemExit(f"Audio not found: {audio_path}")
        mime = guess_mime(audio_path)
        parts.append(types.Part.from_text(text="--- Processed audio clip ---"))
        parts.append(
            types.Part.from_bytes(
                data=audio_path.read_bytes(),
                mime_type=mime,
            )
        )

    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Gemini multimodal mix-description generation (text + optional image + audio).",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Vision/audio-capable Gemini model (default: gemini-2.5-flash).",
    )
    parser.add_argument("--api-key", default=None, help="Override GEMINI_API_KEY / gemini_key.txt.")
    parser.add_argument(
        "--text-data",
        default=None,
        help="Raw parameter text or JSON string. Default: built-in vocal example.",
    )
    parser.add_argument(
        "--params-file",
        default=None,
        help="Path to a file containing mixing parameters (JSON or plain text).",
    )
    parser.add_argument("--image", default=None, help="Optional EQ plot or spectrum image path.")
    parser.add_argument("--audio", default=None, help="Optional processed audio clip path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = load_api_key(args.api_key)
    param_text = load_text_data(args)
    image_path = Path(args.image) if args.image else None
    audio_path = Path(args.audio) if args.audio else None

    client = genai.Client(api_key=api_key)
    user_parts = build_user_parts(param_text, image_path, audio_path)

    print("Model:", args.model)
    print("Inputs: text", end="")
    print(" + image" if image_path else "", end="")
    print(" + audio" if audio_path else "")
    print("-" * 60)

    response = client.models.generate_content(
        model=args.model,
        contents=[types.Content(role="user", parts=user_parts)],
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
    )

    print(response.text or "(empty response)")
    if not response.text:
        sys.exit(1)


if __name__ == "__main__":
    main()
