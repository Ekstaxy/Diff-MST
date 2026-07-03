"""Run text/audio inference on musdb18_processed_v2 samples for listening tests."""

import argparse
import json
import pathlib
import random

import torch
import torchaudio
import yaml

from mst.utils import batch_stereo_peak_normalize, load_diffmst, run_diffmst


def load_mono(path: str, sr: int = 44100) -> torch.Tensor:
    audio, file_sr = torchaudio.load(path, backend="soundfile")
    if file_sr != sr:
        audio = torchaudio.functional.resample(audio, file_sr, sr)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio


def crop_segment(audio: torch.Tensor, start: int, length: int) -> torch.Tensor:
    end = start + length
    if audio.shape[-1] >= end:
        return audio[..., start:end]
    if audio.shape[-1] < length:
        return torch.nn.functional.pad(audio, (0, length - audio.shape[-1]))
    return audio[..., -length:]


def load_json_prompt(path: pathlib.Path, keys: list[str], seed: int | None) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(seed)
    selected_key = rng.choice(keys)

    val = data.get(selected_key, "").strip()
    if val and val != "(no description generated)":
        val = val.rstrip(".")
        if val:
            val = val[0].lower() + val[1:]
        return val
    return "exactly as it is"


def build_prompt(song_dir: pathlib.Path, base_name: str, seed: int | None) -> str:
    keys = ["gain", "pan", "compressor", "eq"]
    vocal_text = load_json_prompt(
        song_dir / f"{base_name}_compare_wet_dry_vocal.json", keys, seed
    )
    inst_text = load_json_prompt(
        song_dir / f"{base_name}_compare_wet_dry_instrumental.json",
        keys,
        seed + 1 if seed is not None else None,
    )
    return f"Make the vocal {vocal_text}, and keep the instrumental {inst_text}."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/models/naive+text.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/work/ajchen2005/musdb18_processed_v2")
    parser.add_argument("--metadata_file", type=str, default="data/musdb18-aug.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--song_name", type=str, default=None)
    parser.add_argument("--aug_idx", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--match_training_crop", action="store_true", default=True)
    parser.add_argument("--no_match_training_crop", dest="match_training_crop", action="store_false")
    parser.add_argument("--window_length", type=int, default=441000)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--length", type=int, default=220500)
    parser.add_argument("--ref_input_type", type=str, default="text", choices=["text", "audio"])
    parser.add_argument("--prompt_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="/work/ajchen2005/inference_train_samples")
    return parser.parse_args()


def save_wav(path: pathlib.Path, audio: torch.Tensor, sr: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(path.as_posix(), audio.cpu(), sr, encoding="PCM_S", bits_per_sample=16)


def run_one_sample(args, model, mix_console, device, song_name: str, aug_idx: int) -> None:
    song_dir = pathlib.Path(args.data_dir) / song_name
    base_name = f"aug_{aug_idx}"

    dry_inst = load_mono((song_dir / "dry" / f"{base_name}_instrumental.wav").as_posix())
    dry_vocal = load_mono((song_dir / "dry" / f"{base_name}_vocal.wav").as_posix())
    ref_inst = load_mono((song_dir / "src_sep" / f"{base_name}_instrumental.wav").as_posix())
    ref_vocal = load_mono((song_dir / "src_sep" / f"{base_name}_vocal.wav").as_posix())
    gt_mix = torchaudio.load((song_dir / f"{base_name}_mix.wav").as_posix(), backend="soundfile")[0]
    if gt_mix.shape[0] > 1:
        gt_mix = gt_mix.mean(dim=0, keepdim=True)

    if args.match_training_crop:
        tracks_inst = crop_segment(dry_inst, args.start_idx, args.window_length)[
            ..., args.window_length // 2 :
        ]
        tracks_vocal = crop_segment(dry_vocal, args.start_idx, args.window_length)[
            ..., args.window_length // 2 :
        ]
        ref_inst_seg = crop_segment(ref_inst, args.start_idx, args.window_length)[
            ..., : args.window_length // 2
        ]
        ref_vocal_seg = crop_segment(ref_vocal, args.start_idx, args.window_length)[
            ..., : args.window_length // 2
        ]
        gt_seg = crop_segment(gt_mix, args.start_idx, args.window_length)[
            ..., args.window_length // 2 :
        ]
        seg_len = args.window_length // 2
    else:
        tracks_inst = crop_segment(dry_inst, args.start_idx, args.length)
        tracks_vocal = crop_segment(dry_vocal, args.start_idx, args.length)
        ref_inst_seg = crop_segment(ref_inst, args.start_idx, args.length)
        ref_vocal_seg = crop_segment(ref_vocal, args.start_idx, args.length)
        gt_seg = crop_segment(gt_mix, args.start_idx, args.length)
        seg_len = args.length

    tracks = torch.cat([tracks_inst, tracks_vocal], dim=0).unsqueeze(0).to(device)
    ref_audio = torch.cat([ref_inst_seg, ref_vocal_seg], dim=0).unsqueeze(0).to(device)

    if args.ref_input_type == "text":
        input_text = build_prompt(song_dir, base_name, args.prompt_seed + aug_idx)
        ref_audio = torch.zeros_like(ref_audio)
    else:
        input_text = None

    with torch.no_grad():
        pred_mix, pred_mixed_tracks, _, _, _ = run_diffmst(
            tracks,
            ref_audio,
            model,
            mix_console,
            text=input_text,
            use_master_bus=False,
        )

    pred_mix = batch_stereo_peak_normalize(pred_mix)
    gt_stereo = batch_stereo_peak_normalize(gt_seg.repeat(2, 1).unsqueeze(0))

    out_dir = pathlib.Path(args.output_dir) / args.split / song_name / base_name
    save_wav(out_dir / "pred_mix.wav", pred_mix.squeeze(0))
    save_wav(out_dir / "gt_mix.wav", gt_stereo.squeeze(0))
    save_wav(out_dir / "dry_inst.wav", tracks_inst)
    save_wav(out_dir / "dry_vocal.wav", tracks_vocal)

    with open(out_dir / "prompt.txt", "w", encoding="utf-8") as f:
        f.write(input_text or "(audio reference)")

    meta = {
        "song_name": song_name,
        "base_name": base_name,
        "split": args.split,
        "ref_input_type": args.ref_input_type,
        "segment_length": seg_len,
        "prompt": input_text,
        "checkpoint": args.checkpoint,
        "config": args.config,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {out_dir}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.metadata_file, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    song_names = [args.song_name] if args.song_name else metadata.get(args.split, [])[: args.num_samples]

    model, mix_console = load_diffmst(args.config, args.checkpoint, map_location=device)
    model.to(device).eval()
    mix_console.to(device).eval()

    for song_name in song_names:
        run_one_sample(args, model, mix_console, device, song_name, args.aug_idx)


if __name__ == "__main__":
    main()
