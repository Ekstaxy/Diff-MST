import os
import glob
import yaml
import torch
import torchaudio
import argparse
import random
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mst.modules import AdvancedMixConsole, RoFormerRemixer
from mst.mixing import naive_random_mix

from audio_analysis import run_aug_analysis

EXPECTED_STEMS = ["vocals.wav", "bass.wav", "drums.wav", "other.wav"]


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-process dataset: Mix -> Separate -> Analyze -> Describe")
    parser.add_argument("--config", type=str, default="configs/data/musdb18-2.yaml", help="Path to data config")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the processed dataset")
    parser.add_argument("--augmentations", type=int, default=10, help="How many random mixes per song?")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--duration", type=float, default=20.0, help="Duration in seconds per clip")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--roformer_model", type=str, default="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    parser.add_argument("--max_songs", type=int, default=None, help="Limit number of songs (from beginning)")
    parser.add_argument("--start_song_idx", type=int, default=0, help="Start index for processing songs")
    parser.add_argument("--end_song_idx", type=int, default=None, help="End index (exclusive) for processing songs")
    parser.add_argument("--force_text", action="store_true", help="Re-run EQ/metrics/Gemini even if outputs exist")
    parser.add_argument("--gemini_model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--gemini_api_key", type=str, default=None)
    parser.add_argument(
        "--track_root_dirs",
        nargs="+",
        default=None,
        help="Override MUSDB root dirs (used if not in config YAML)",
    )
    return parser.parse_args()


def load_metadata(config_path):
    config_path = os.path.join(os.path.dirname(__file__), "..", config_path) if not os.path.isabs(config_path) else config_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    try:
        init_args = config["data"]["init_args"]
        track_root_dirs = init_args.get("track_root_dirs", [])
        metadata_files = init_args.get("metadata_files", [])
    except KeyError:
        track_root_dirs = config.get("track_root_dirs", [])
        metadata_files = config.get("metadata_files", [])
    return track_root_dirs, metadata_files


def get_song_dirs(track_root_dirs, metadata_files=None):
    all_song_paths = []
    expected_stems = set(EXPECTED_STEMS)
    if isinstance(track_root_dirs, str):
        track_root_dirs = [track_root_dirs]
    for root_dir in track_root_dirs:
        for root, _dirs, files in os.walk(root_dir):
            if expected_stems.issubset(set(files)):
                all_song_paths.append(root)
    return sorted(list(set(all_song_paths)))


def ensure_stereo(audio: torch.Tensor) -> torch.Tensor:
    """(C, T) with C in {1, 2}."""
    if audio.shape[0] == 1:
        return audio.repeat(2, 1)
    if audio.shape[0] > 2:
        return audio[:2]
    return audio


def to_mono_track(stereo: torch.Tensor) -> torch.Tensor:
    """(2, T) -> (1, T) for AdvancedMixConsole input."""
    return stereo.mean(dim=0, keepdim=True)


def save_audio(tensor: torch.Tensor, path: str, sr: int) -> None:
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    t = tensor
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.shape[0] == 1:
        t = t.repeat(2, 1)
    data = t.numpy().T
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, data, sr)


def build_aug_paths(save_dir: Path, aug_idx: int) -> dict[str, Path]:
    p = f"aug_{aug_idx}"
    dry = save_dir / "dry"
    wet = save_dir / "wet"
    src_sep = save_dir / "src_sep"
    return {
        "dry_vocal": dry / f"{p}_vocal.wav",
        "dry_instrumental": dry / f"{p}_instrumental.wav",
        "wet_vocal": wet / f"{p}_vocal.wav",
        "wet_instrumental": wet / f"{p}_instrumental.wav",
        "src_sep_vocal": src_sep / f"{p}_vocal.wav",
        "src_sep_instrumental": src_sep / f"{p}_instrumental.wav",
        "mix": save_dir / f"{p}_mix.wav",
        "params": save_dir / f"{p}_params.pt",
        "dry_vocal_txt": dry / f"{p}_vocal.txt",
        "dry_instrumental_txt": dry / f"{p}_instrumental.txt",
        "wet_vocal_txt": wet / f"{p}_vocal.txt",
        "wet_instrumental_txt": wet / f"{p}_instrumental.txt",
        "src_sep_vocal_txt": src_sep / f"{p}_vocal.txt",
        "src_sep_instrumental_txt": src_sep / f"{p}_instrumental.txt",
        "mix_txt": save_dir / f"{p}_mix.txt",
    }


def aug_audio_complete(paths: dict[str, Path]) -> bool:
    return paths["mix"].is_file() and paths["dry_vocal"].is_file()


def run_text_pipeline(
    save_dir: Path,
    aug_idx: int,
    paths: dict[str, Path],
    args,
) -> None:
    wav_for_metrics = {
        "dry_vocal": paths["dry_vocal"],
        "dry_instrumental": paths["dry_instrumental"],
        "wet_vocal": paths["wet_vocal"],
        "wet_instrumental": paths["wet_instrumental"],
        "src_sep_vocal": paths["src_sep_vocal"],
        "src_sep_instrumental": paths["src_sep_instrumental"],
        "mix": paths["mix"],
    }
    metrics = run_aug_analysis(
        save_dir,
        aug_idx,
        wav_for_metrics,
        args.sample_rate,
        force=args.force_text,
    )
    from gemini_descriptions import run_aug_gemini

    run_aug_gemini(
        save_dir,
        aug_idx,
        paths,
        metrics,
        model=args.gemini_model,
        api_key=args.gemini_api_key,
        force=args.force_text,
    )


def process_song(song_dir, args, mixer, separator, output_root):
    song_name = os.path.basename(song_dir)
    save_dir = Path(output_root) / song_name
    save_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("dry", "wet", "src_sep", "eq"):
        (save_dir / sub).mkdir(parents=True, exist_ok=True)

    existing_mixes = glob.glob(str(save_dir / "aug_*_mix.wav"))
    if len(existing_mixes) >= args.augmentations and not args.force_text:
        all_have_metrics = all(
            (save_dir / f"aug_{i}_metrics.json").is_file() for i in range(args.augmentations)
        )
        if all_have_metrics:
            return

    start_aug_idx = 0
    if existing_mixes:
        indices = []
        for f in existing_mixes:
            base = os.path.basename(f)
            try:
                indices.append(int(base.split("_")[1]))
            except (IndexError, ValueError):
                pass
        if indices:
            start_aug_idx = max(indices) + 1

    loaded_stems = {}
    length_samples = int(args.duration * args.sample_rate)
    max_len = 0

    for stem_name in EXPECTED_STEMS:
        file_path = os.path.join(song_dir, stem_name)
        if os.path.exists(file_path):
            audio, sr = torchaudio.load(file_path)
            if sr != args.sample_rate:
                audio = torchaudio.transforms.Resample(sr, args.sample_rate)(audio)
            audio = ensure_stereo(audio)
            loaded_stems[stem_name] = audio
            max_len = max(max_len, audio.shape[-1])

    if "vocals.wav" not in loaded_stems:
        print(f"Skipping {song_name}: No vocals.wav found.")
        return

    if max_len < length_samples:
        print(f"Skipping {song_name}: too short.")
        return

    target_total = args.augmentations

    for i in range(target_total):
        paths = build_aug_paths(save_dir, i)
        metrics_path = save_dir / f"aug_{i}_metrics.json"

        if aug_audio_complete(paths) and metrics_path.is_file() and not args.force_text:
            continue

        if aug_audio_complete(paths) and not metrics_path.is_file():
            try:
                run_text_pipeline(save_dir, i, paths, args)
            except Exception as e:
                print(f"Text pipeline failed for {song_name} aug {i}: {e}")
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "resource" in err_str:
                    print("API Quota exceeded. Exiting...")
                    import sys; sys.exit(1)
            continue

        if i < start_aug_idx and aug_audio_complete(paths):
            if not metrics_path.is_file():
                try:
                    run_text_pipeline(save_dir, i, paths, args)
                except Exception as e:
                    print(f"Text pipeline failed for {song_name} aug {i}: {e}")
                    err_str = str(e).lower()
                    if "429" in err_str or "quota" in err_str or "resource" in err_str:
                        print("API Quota exceeded. Exiting...")
                        import sys; sys.exit(1)
            continue

        if i < start_aug_idx:
            continue

        valid_crop = False
        start_idx = 0
        vocal_track = loaded_stems["vocals.wav"]

        for _ in range(50):
            start = random.randint(0, max_len - length_samples)
            voc_slice = vocal_track[:, start : start + length_samples]
            rms = torch.sqrt(torch.mean(voc_slice**2))
            dbfs = 20 * torch.log10(rms + 1e-8)
            if dbfs > -35.0:
                valid_crop = True
                start_idx = start
                break

        if not valid_crop:
            print(f"Warning: Could not find valid vocal segment for {song_name} aug {i}. Skipping.")
            continue

        slices = []
        for stem_name in EXPECTED_STEMS:
            if stem_name in loaded_stems:
                slices.append(loaded_stems[stem_name][:, start_idx : start_idx + length_samples])
            else:
                slices.append(torch.zeros(2, length_samples))

        dry_vocal = slices[0]
        # Instrumental bus: sum bass + drums + other BEFORE any mix-console processing
        dry_inst_bus = slices[1] + slices[2] + slices[3]

        vocal_mono = to_mono_track(dry_vocal).to(args.device)
        inst_mono = to_mono_track(dry_inst_bus).to(args.device)
        batch_slice = torch.cat([vocal_mono, inst_mono], dim=0).unsqueeze(0)

        (
            mixed_tracks,
            ref_mix,
            _track_param_dict,
            _fx_bus_param_dict,
            _master_bus_param_dict,
            mix_params,
            fx_bus_params,
            master_bus_params,
        ) = naive_random_mix(
            batch_slice,
            mixer,
            use_track_input_fader=True,
            use_track_panner=True,
            use_track_eq=True,
            use_track_compressor=True,
            use_fx_bus=False,
            use_master_bus=False,
            use_ouput_fader=False,  # keep mix = sum(wet tracks), no master-only gain
        )

        # mixed_tracks: (batch, stereo_L/R, num_tracks, time) after stereo_panner
        wet_vocal = mixed_tracks[0, :, 0, :].cpu()
        wet_instrumental = mixed_tracks[0, :, 1, :].cpu()
        # Mix is exactly wet vocal + wet instrumental (no peak/LUFS normalization)
        wet_mix = wet_vocal + wet_instrumental

        bus_diff = (ref_mix[0].cpu() - wet_mix).abs().max().item()
        if bus_diff > 1e-5:
            print(
                f"Warning: {song_name} aug {i}: master_bus vs wet sum max diff {bus_diff:.2e}"
            )

        try:
            with torch.no_grad():
                separated_sources = separator(wet_mix.unsqueeze(0).to(args.device))
        except Exception as e:
            print(f"Separation failed for {song_name} aug {i}: {e}")
            torch.cuda.empty_cache()
            continue

        src_sep_vocal = separated_sources[0, 1].cpu()
        src_sep_instrumental = separated_sources[0, 0].cpu()

        save_audio(dry_vocal, str(paths["dry_vocal"]), args.sample_rate)
        save_audio(dry_inst_bus, str(paths["dry_instrumental"]), args.sample_rate)
        save_audio(wet_vocal, str(paths["wet_vocal"]), args.sample_rate)
        save_audio(wet_instrumental, str(paths["wet_instrumental"]), args.sample_rate)
        save_audio(wet_mix, str(paths["mix"]), args.sample_rate)
        save_audio(src_sep_vocal, str(paths["src_sep_vocal"]), args.sample_rate)
        save_audio(src_sep_instrumental, str(paths["src_sep_instrumental"]), args.sample_rate)

        torch.save(
            {
                "crop_start": start_idx,
                "track_params": mix_params[0].cpu(),
                "fx_bus_params": fx_bus_params[0].cpu(),
                "master_bus_params": master_bus_params[0].cpu(),
                "note": "mix = wet_vocal + wet_instrumental (no loudness norm); src_sep from that mix",
            },
            paths["params"],
        )

        del separated_sources
        torch.cuda.empty_cache()

        try:
            run_text_pipeline(save_dir, i, paths, args)
        except Exception as e:
            print(f"Text pipeline failed for {song_name} aug {i}: {e}")
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "resource" in err_str:
                print("API Quota exceeded. Exiting...")
                import sys; sys.exit(1)
            import traceback

            traceback.print_exc()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    print("Setting up dataset generation...")
    print(f"Output Dir: {args.output_dir}")
    print(f"Augmentations per song: {args.augmentations}")
    if args.start_song_idx > 0 or args.end_song_idx is not None:
        print(f"Song Index Range: {args.start_song_idx} to {args.end_song_idx}")
    elif args.max_songs is not None:
        print(f"Max songs: {args.max_songs}")

    mixer = AdvancedMixConsole(sample_rate=args.sample_rate).to(args.device)
    separator = RoFormerRemixer(sample_rate=args.sample_rate, model_name=args.roformer_model).to(args.device)
    separator.eval()

    if args.track_root_dirs:
        track_root_dirs = args.track_root_dirs
    else:
        track_root_dirs, _metadata_files = load_metadata(args.config)
    if isinstance(track_root_dirs, str):
        track_root_dirs = [track_root_dirs]
    if not track_root_dirs:
        track_root_dirs = ["/work/ajchen2005/musdb18hq"]
        print(f"No track_root_dirs in config; using default: {track_root_dirs[0]}")
    song_dirs = get_song_dirs(track_root_dirs)
    
    # Handle song indexing/filtering
    if args.start_song_idx > 0 or args.end_song_idx is not None:
        song_dirs = song_dirs[args.start_song_idx:args.end_song_idx]
    elif args.max_songs is not None:
        song_dirs = song_dirs[: args.max_songs]

    print(f"Found {len(song_dirs)} songs")

    for song_dir in tqdm(song_dirs, desc="Processing Songs"):
        try:
            process_song(song_dir, args, mixer, separator, args.output_dir)
        except KeyboardInterrupt:
            print("Interrupted by user.")
            break
        except Exception as e:
            print(f"Error processing {song_dir}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
