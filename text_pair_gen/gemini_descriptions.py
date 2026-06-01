"""Gemini API helpers for single-track and pairwise mix descriptions."""

from __future__ import annotations

import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEY_FILE = REPO_ROOT / "gemini_key.txt"

"""\
You are a professional mix engineer explaining how a track sounds to a musician.
Translate the provided audio metrics and EQ image into a natural, intuitive description.

CRITICAL RULES:
1. NO NUMBERS: Do not output exact frequencies (Hz), volumes (dB, LUFS), or metric values.
2. EXTREME BREVITY: Restrict your response to 1 to 2 short sentences (maximum 30 words).
3. VOCABULARY: Use expressive mixing adjectives and adverbs (e.g., slightly muffled, overly harsh, warm, punchy, airy, boomy, tinny, crisp).
4. VIBE: You may use combinations or imagery (e.g., "soft yet vibrant," "distant and hollow") to convey the feel.
5. FOCUS: Describe the perceptual acoustic feel (low-end body, top-end air, presence, stereo width, dynamics) rather than summarizing the raw data.\
"""

SINGLE_TRACK_SYSTEM = """\
You are a professional mix engineer explaining how a track sounds to a musician.
Translate the provided audio metrics and EQ image into a natural, intuitive description.

When an EQ curve image is provided, relate the visual shape to tonal changes.
When an audio clip is provided, relate the heard sound to the metrics and role label.

CRITICAL RULES:
1. NO NUMBERS: Do not output exact frequencies (Hz), volumes (dB, LUFS), or metric values.
2. EXTREME BREVITY: Restrict your response to 1 to 2 short sentences (maximum 20 words).
3. VOCABULARY: Use expressive mixing adjectives and adverbs (e.g., slightly muffled, overly harsh, warm, punchy, airy, boomy, tinny, crisp).
4. VIBE: You may use combinations or imagery (e.g., "soft yet vibrant," "distant and hollow") to convey the feel.
5. FOCUS: Describe the perceptual acoustic feel (low-end body, top-end air, presence, stereo width, dynamics) rather than summarizing the raw data.
\
"""

PAIRWISE_SYSTEM = """\
You are an expert mix engineer comparing two versions of the same track role
(e.g. dry vs wet, or source-separated vs dry).

You will receive two audio clips, two EQ spectrum images, and JSON metrics for each.
Describe how the SECOND clip (B) differs from the FIRST clip (A) in perceptual mixing terms.

The second clip (B) is the variant, and the first clip (A) is the reference.

The second clip is only processed with gain, pan, compressor, and EQ. No other effects are applied.

CRITICAL RULES:
1. NO ARTIFACT WORDS: NEVER use the words "Clip A", "Clip B", "reference", or "variant" in your output.
2. NO NUMBERS: Do not use exact frequencies (Hz), ratios, or levels (dB, LUFS).
3. RELATIVE PHRASING: Describe the variant as a relative change or a target state compared to the baseline (e.g., "A warmer and softer vocal", "Make the guitar quieter", "Panned further right").
4. EXTREME BREVITY: Limit each field to a short, direct phrase (maximum 10 words per field).
5. VOCABULARY: Use concrete mixing adjectives and relative modifiers (e.g., brighter, warmer, punchier, less muddy, wider, more upfront, heavily compressed).

Respond with ONLY valid JSON (no markdown fences) using exactly these five keys, describing the processed variant's state:
- "overall": holistic difference (e.g., "A more distant and hollow sound")
- "gain": loudness/level (e.g., "Quieter and pushed further back in the mix")
- "pan": stereo placement (e.g., "Panned hard right")
- "compressor": dynamics (e.g., "Heavily squashed with smoothed transients")
- "eq": tonal balance (e.g., "Thinner with less low-mid warmth", "Darker vocal with reduced high-end shimmer")
...
VOCABULARY TO USE:
- EQ: warm, muddy, boomy, thin, boxy, harsh, bright, airy, crisp, dark, muffled, tinny.
- Gain: upfront, prominent, distant, buried, quiet, loud.
- Compressor: punchy, snappy, smooth, glued, squashed, pumping, flat, tight, dynamic.
- Pan: centered, wide, narrow, panned hard left, panned right.
...\
"""

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

COMPARE_KEYS = ("overall", "gain", "pan", "compressor", "eq")

MAX_RETRIES = 5
RETRY_DELAY_SEC = 2.0


def load_api_key(explicit: str | None = None) -> str:
    if explicit:
        return explicit.strip()
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()
    if DEFAULT_KEY_FILE.is_file():
        return DEFAULT_KEY_FILE.read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "No API key: set GEMINI_API_KEY, pass --gemini_api_key, or create gemini_key.txt at repo root."
    )


def guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in MIME_OVERRIDES:
        return MIME_OVERRIDES[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    raise ValueError(f"Could not infer MIME type for {path}")


def _file_part(path: Path, label: str) -> list[types.Part]:
    if not path.is_file():
        raise FileNotFoundError(path)
    mime = guess_mime(path)
    return [
        types.Part.from_text(text=f"--- {label} ---"),
        types.Part.from_bytes(data=path.read_bytes(), mime_type=mime),
    ]


def _generate(
    client: genai.Client,
    model: str,
    system_instruction: str,
    user_parts: list[types.Part],
    *,
    json_mode: bool = False,
) -> str:
    config_kw: dict[str, Any] = {
        "system_instruction": system_instruction,
        "max_output_tokens": 200,  # 加上最大的輸出限制來避免模型偷水字數
    }
    if json_mode:
        config_kw["response_mime_type"] = "application/json"

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=user_parts)],
                config=types.GenerateContentConfig(**config_kw),
            )
            text = response.text or ""
            if not text.strip():
                raise RuntimeError("Empty Gemini response")
            return text.strip()
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if "429" in err_str or "resource" in err_str or "quota" in err_str:
                time.sleep(RETRY_DELAY_SEC * (attempt + 1))
                continue
            elif "empty gemini response" in err_str:
                time.sleep(RETRY_DELAY_SEC * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"Gemini failed after {MAX_RETRIES} retries: {last_err}")


def describe_single_track(
    client: genai.Client,
    model: str,
    *,
    role_label: str,
    audio_path: Path,
    eq_plot_path: Path,
    metrics: dict[str, Any],
) -> str:
    metrics_text = json.dumps(metrics, indent=2)
    parts: list[types.Part] = [
        types.Part.from_text(
            text=(
                f"Describe this audio in natural mixing language.\n"
                f"Track role: {role_label}\n\n"
                f"--- Analysis metrics (JSON) ---\n{metrics_text}\n"
            )
        ),
    ]
    parts.extend(_file_part(eq_plot_path, "EQ / spectrum image"))
    parts.extend(_file_part(audio_path, "Audio clip"))

    return _generate(client, model, SINGLE_TRACK_SYSTEM, parts, json_mode=False)


def describe_pairwise(
    client: genai.Client,
    model: str,
    *,
    comparison_label: str,
    audio_a: Path,
    audio_b: Path,
    eq_a: Path,
    eq_b: Path,
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
) -> dict[str, str]:
    parts: list[types.Part] = [
        types.Part.from_text(
            text=(
                f"Compare clip B relative to clip A.\n"
                f"Comparison: {comparison_label}\n"
                f"A is the reference (typically dry); B is the variant.\n\n"
                f"--- Metrics A ---\n{json.dumps(metrics_a, indent=2)}\n\n"
                f"--- Metrics B ---\n{json.dumps(metrics_b, indent=2)}\n"
            )
        ),
    ]
    parts.extend(_file_part(audio_a, "Audio A"))
    parts.extend(_file_part(eq_a, "EQ image A"))
    parts.extend(_file_part(audio_b, "Audio B"))
    parts.extend(_file_part(eq_b, "EQ image B"))

    raw = _generate(client, model, PAIRWISE_SYSTEM, parts, json_mode=True)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
        else:
            raise

    result = {k: str(data.get(k, "")).strip() for k in COMPARE_KEYS}
    for k in COMPARE_KEYS:
        if not result[k]:
            result[k] = "(no description generated)"
    return result


def run_aug_gemini(
    song_dir: Path,
    aug_idx: int,
    paths: dict[str, Path],
    metrics: dict[str, Any],
    *,
    model: str,
    api_key: str | None = None,
    force: bool = False,
) -> None:
    """Run 7 single-track + 4 pairwise Gemini calls for one augmentation."""
    song_dir = Path(song_dir)
    client = genai.Client(api_key=load_api_key(api_key))
    prefix = f"aug_{aug_idx}"

    single_specs = [
        ("dry_vocal", paths["dry_vocal"], paths["dry_vocal_txt"], "dry vocal (unprocessed)"),
        ("dry_instrumental", paths["dry_instrumental"], paths["dry_instrumental_txt"], "dry instrumental bus (bass+drums+other sum, unprocessed)"),
        ("wet_vocal", paths["wet_vocal"], paths["wet_vocal_txt"], "wet vocal (after random mix console processing)"),
        ("wet_instrumental", paths["wet_instrumental"], paths["wet_instrumental_txt"], "wet instrumental (after random mix console processing)"),
        ("src_sep_vocal", paths["src_sep_vocal"], paths["src_sep_vocal_txt"], "source-separated vocal from final mix"),
        ("src_sep_instrumental", paths["src_sep_instrumental"], paths["src_sep_instrumental_txt"], "source-separated instrumental from final mix"),
        ("mix", paths["mix"], paths["mix_txt"], "full stereo mix (wet vocal + wet instrumental)"),
    ]

    for metric_key, audio_path, txt_path, role in single_specs:
        if txt_path.is_file() and not force:
            continue
        eq_plot = Path(metrics[metric_key]["eq_plot"])
        text = describe_single_track(
            client,
            model,
            role_label=role,
            audio_path=audio_path,
            eq_plot_path=eq_plot,
            metrics={k: v for k, v in metrics[metric_key].items() if k != "eq_plot"},
        )
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(text, encoding="utf-8")

    def eq_for(key: str) -> Path:
        return Path(metrics[key]["eq_plot"])

    def met_for(key: str) -> dict[str, Any]:
        return {k: v for k, v in metrics[key].items() if k != "eq_plot"}

    pairwise_specs = [
        (
            f"{prefix}_compare_wet_dry_vocal.json",
            "wet vocal vs dry vocal",
            paths["dry_vocal"],
            paths["wet_vocal"],
            "dry_vocal",
            "wet_vocal",
        ),
        (
            f"{prefix}_compare_wet_dry_instrumental.json",
            "wet instrumental vs dry instrumental",
            paths["dry_instrumental"],
            paths["wet_instrumental"],
            "dry_instrumental",
            "wet_instrumental",
        ),
        (
            f"{prefix}_compare_srcsep_dry_vocal.json",
            "source-separated vocal vs dry vocal",
            paths["dry_vocal"],
            paths["src_sep_vocal"],
            "dry_vocal",
            "src_sep_vocal",
        ),
        (
            f"{prefix}_compare_srcsep_dry_instrumental.json",
            "source-separated instrumental vs dry instrumental",
            paths["dry_instrumental"],
            paths["src_sep_instrumental"],
            "dry_instrumental",
            "src_sep_instrumental",
        ),
    ]

    for out_name, label, path_a, path_b, key_a, key_b in pairwise_specs:
        out_path = song_dir / out_name
        if out_path.is_file() and not force:
            continue
        result = describe_pairwise(
            client,
            model,
            comparison_label=label,
            audio_a=path_a,
            audio_b=path_b,
            eq_a=eq_for(key_a),
            eq_b=eq_for(key_b),
            metrics_a=met_for(key_a),
            metrics_b=met_for(key_b),
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
