import torch
import pathlib
import argparse
import torchaudio
from tqdm import tqdm
import pyloudnorm as pyln

from mst.utils import load_diffmst, run_diffmst

def parse_args():
    parser = argparse.ArgumentParser(description='Generate audio examples for listening test')
    # Model configs
    parser.add_argument("--config", type=str, required=True,
                        help='Path to naive.yaml')
    parser.add_argument("--checkpoint", type=str, required=True,
                        help='Path to model checkpoint')

    
    # Audio parameters
    parser.add_argument("--tracks_path", type=str, 
                        default="/mnt/gestalt/home/rakec/data/diff-mst/MedleyDB_V2/V2/TleilaxEnsemble_Late/TleilaxEnsemble_Late_RAW",
                        help='Path to folder with audio tracks to use for mixing')
    
    # Looping parameters
    parser.add_argument("--control_type", type=str, nargs='+', default=["audio"],
                        help="Control types to use for mixing (audio or text)")    
    parser.add_argument("--control_info", type=str, nargs='+', 
                        default=["/kaggle/input/medley-db-v2/V2/TleilaxEnsemble_Late/TleilaxEnsemble_Late_MIX.wav", (-1, 1, "The sound is dark"), (2, 1, "The sound is dark")],
                        help="Control information (file paths for audio, text prompts for text in format: (track, weight, 'text'). If track is -1, use master bus.)")
    
    # Verse/Chorus indices
    parser.add_argument('--track-verse-idx', type=int, required=True,
                        help='Track verse start index (samples)')
    parser.add_argument('--track-chorus-idx', type=int, required=True,
                        help='Track chorus start index (samples)')
    parser.add_argument('--ref-verse-idx', type=int, required=True,
                        help='Reference verse start index (samples)')
    parser.add_argument('--ref-chorus-idx', type=int, required=True,
                        help='Reference chorus start index (samples)')
    
    # Other parameters
    parser.add_argument("--output_dir", type=str, default="./eval_outputs",
                        help='Directory to save generated audio examples')
    parser.add_argument("--exp_name", type=str, default="test", 
                        help='Experiment name for output folder')
    parser.add_argument('--target_lufs', type=float, default=-22.0,
                        help='Target output LUFS')

    parser.add_argument('--sum_only', type=bool, default=False,
                        help='Whether to only run the sum baseline')

    return parser.parse_args()

def equal_loudness_mix(tracks: torch.Tensor, *args, **kwargs):

    meter = pyln.Meter(44100)
    target_lufs_db = -48.0

    norm_tracks = []
    for track_idx in range(tracks.shape[1]):
        track = tracks[:, track_idx : track_idx + 1, :]
        lufs_db = meter.integrated_loudness(track.squeeze(0).permute(1, 0).numpy())

        if lufs_db < -80.0:
            print(f"Skipping track {track_idx} with {lufs_db:.2f} LUFS.")
            continue

        lufs_delta_db = target_lufs_db - lufs_db
        track *= 10 ** (lufs_delta_db / 20)
        norm_tracks.append(track)

    norm_tracks = torch.cat(norm_tracks, dim=1)
    # create a sum mix with equal loudness
    sum_mix = torch.sum(norm_tracks, dim=1, keepdim=True).repeat(1, 2, 1)
    sum_mix /= sum_mix.abs().max()

    return sum_mix, None, None, None

def main():
    args = parse_args()

    meter = pyln.Meter(44100)
    target_lufs_db = args.target_lufs
    output_dir = pathlib.Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = {
        "diffmst":{
            "model": load_diffmst(
                config_path=args.config,
                ckpt_path=args.checkpoint
            ),
            "func": run_diffmst
        },
        "sum": {
            "model": (None, None),
            "func": equal_loudness_mix,
        }
    }

    # Find all tracks
    track_filepaths = list(pathlib.Path(args.tracks_path).glob("*.wav"))
    print(f"[INFO] Found {len(track_filepaths)} tracks in {args.tracks_path}")

    tracks = []
    lengths = []
    for track_idx, track_filepath in enumerate(track_filepaths):
        audio, sr = torchaudio.load(track_filepath, backend="soundfile")

        if sr != 44100:
            audio = torchaudio.functional.resample(audio, sr, 44100)

        if audio.shape[0] == 2:
            audio = audio.mean(dim=0, keepdim=True)

        chs, seq_len = audio.shape

        for ch_idx in range(chs):
            tracks.append(audio[ch_idx : ch_idx + 1, :])
            lengths.append(audio.shape[-1])

    # Find max length and pad if shorter
    max_length = max(lengths)
    for track_idx in range(len(tracks)):
        tracks[track_idx] = torch.nn.functional.pad(
            tracks[track_idx], (0, max_length - lengths[track_idx])
        )
    
    tracks = torch.cat(tracks, dim=0)
    tracks = tracks.view(1, -1, max_length)

    print(f"[INFO] Tracks shape: {tracks.shape}")

    if args.sum_only:
        for song_section in ["verse", "chorus"]:
            print(f"[INFO] Mixing {song_section} with sum baseline...")
            if song_section == "verse":
                track_start_idx = args.track_verse_idx
            else:
                track_start_idx = args.track_chorus_idx

            mix_tracks = tracks[..., track_start_idx : track_start_idx + (44100 * 10)]

            sum_mix, _, _, _ = equal_loudness_mix(mix_tracks)

            mix_lufs_db = meter.integrated_loudness(
                sum_mix.squeeze(0).permute(1, 0).numpy()
            )
            lufs_delta_db = target_lufs_db - mix_lufs_db
            sum_mix = sum_mix * 10 ** (lufs_delta_db / 20)

            mix_filepath = output_dir / f"sum-baseline-{song_section}-lufs-{int(target_lufs_db)}.wav"
            torchaudio.save(mix_filepath, sum_mix.view(2, -1), 44100)
        return
        

    assert len(args.control_type) == len(args.control_info), \
        "Number of control types must match number of control info entries."
    for c_idx, c_type in enumerate(args.control_type):
        assert c_type in ["audio", "text"], f"Unsupported control type: {c_type}"
        if c_type == "audio":
            example = {
                "tracks": args.tracks_path,
                "track_verse_start_idx": args.track_verse_idx,
                "track_chorus_start_idx": args.track_chorus_idx,
                "ref": args.control_info[c_idx],
                "ref_verse_start_idx": args.ref_verse_idx,
                "ref_chorus_start_idx": args.ref_chorus_idx
            }

            ref_audio, ref_sr = torchaudio.load(example["ref"], backend="soundfile")
            if ref_sr != 44100:
                ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 44100)
            ref_audio = ref_audio.view(1, 2, -1)
            print(f"[INFO] reference audio shape: {ref_audio.shape}")

            for song_section in ["verse", "chorus"]:
                print(f"[INFO] Mixing {song_section}...")
                if song_section == "verse":
                    track_start_idx = example["track_verse_start_idx"]
                    ref_start_idx = example["ref_verse_start_idx"]
                else:
                    track_start_idx = example["track_chorus_start_idx"]
                    ref_start_idx = example["ref_chorus_start_idx"]

                if track_start_idx + 44100 * 10 > tracks.shape[-1]:
                    print(f"[Warning] Tracks too short for this section.")
                if ref_start_idx + 44100 * 10 > ref_audio.shape[-1]:
                    print(f"[Warning] Reference too short for this section.")

                mix_tracks = tracks
                mix_tracks = tracks[..., track_start_idx : track_start_idx + (44100 * 10 * 2)]
                track_start_idx = 0


                method_name = "diffmst"
                method = methods[method_name]
                print(f"[INFO] Applying method: {method_name}")


                model, mix_console = method["model"]
                model = model.to("cpu") if model is not None else None
                mix_console = mix_console.to("cpu") if mix_console is not None else None
                func = method["func"]

                with torch.no_grad():
                    result = func(
                        mix_tracks.clone(),
                        ref_audio.clone(),
                        model,
                        mix_console,
                        track_start_idx=track_start_idx,
                        ref_start_idx=ref_start_idx,
                    )

                    (
                        pred_mix,
                        pred_mixed_tracks,
                        pred_track_param_dict,
                        pred_fx_bus_param_dict,
                        pred_master_bus_param_dict,
                    ) = result

                    bs, chs, seq_len = pred_mix.shape

                    mix_lufs_db = meter.integrated_loudness(
                        pred_mix.squeeze(0).permute(1, 0).numpy()
                    )
                    lufs_delta_db = target_lufs_db - mix_lufs_db
                    pred_mix = pred_mix * 10 ** (lufs_delta_db / 20)

                    mix_filepath = output_dir / f"step{c_idx}-{method_name}-ref={song_section}.wav"
                    torchaudio.save(mix_filepath, pred_mix.view(chs, -1), 44100)
                    
        elif c_type == "text":
            text = args.control_info[c_idx]
            print(f"[INFO] Using text prompt: {text[2]}, weight: {text[1]}, track: {text[0]}")
            
            example = {
                "tracks": args.tracks_path,
                "track_verse_start_idx": args.track_verse_idx,
                "track_chorus_start_idx": args.track_chorus_idx,
                "ref": args.control_info[c_idx],
                "ref_verse_start_idx": args.ref_verse_idx,
                "ref_chorus_start_idx": args.ref_chorus_idx
            }

            num_tracks = pred_mixed_tracks.shape[2]     # pred_mixed_tracks: (bs, 2, num_tracks, seq_len)
            if example["ref"][0] < -1 or example["ref"][0] >= num_tracks:
                raise ValueError(f"Invalid track index {example['ref'][0]} for {num_tracks} tracks.")

            if example["ref"][0] == -1:
                ref_audio = pred_mix
            else:
                ref_audio = pred_mixed_tracks
                ref_audio = ref_audio.view(1, 2*num_tracks, -1)

            print(f"[INFO] reference audio shape: {ref_audio.shape}")
            
            prev_fx_bus_param_dict = pred_fx_bus_param_dict
            prev_track_param_dict = pred_track_param_dict
            prev_master_bus_param_dict = pred_master_bus_param_dict

            for song_section in ["verse", "chorus"]:
                print(f"[INFO] Mixing {song_section}...")
                if song_section == "verse":
                    track_start_idx = example["track_verse_start_idx"]
                    ref_start_idx = example["ref_verse_start_idx"]
                else:
                    track_start_idx = example["track_chorus_start_idx"]
                    ref_start_idx = example["ref_chorus_start_idx"]

                if track_start_idx + 44100 * 10 > tracks.shape[-1]:
                    print(f"[Warning] Tracks too short for this section.")
                if ref_start_idx + 44100 * 10 > ref_audio.shape[-1]:
                    print(f"[Warning] Reference too short for this section.")


                method_name = "diffmst"
                method = methods[method_name]
                print(f"[INFO] Applying method: {method_name}")


                model, mix_console = method["model"]
                model = model.to("cpu") if model is not None else None
                mix_console = mix_console.to("cpu") if mix_console is not None else None
                func = method["func"]
                

                with torch.no_grad():
                    result = func(
                        mix_tracks.clone(),
                        ref_audio.clone(),
                        model,
                        mix_console,
                        text=example["ref"],
                        track_start_idx=track_start_idx,
                        ref_start_idx=ref_start_idx,
                        prev_fx_bus_param_dict=prev_fx_bus_param_dict,
                        prev_master_bus_param_dict=prev_master_bus_param_dict,
                        prev_track_param_dict=prev_track_param_dict,
                    )

                    (
                        pred_mix,
                        pred_mixed_tracks,
                        pred_track_param_dict,
                        pred_fx_bus_param_dict,
                        pred_master_bus_param_dict,
                    ) = result
                    
                    
                    
                    bs, chs, seq_len = pred_mix.shape

                    mix_lufs_db = meter.integrated_loudness(
                        pred_mix.squeeze(0).permute(1, 0).numpy()
                    )
                    lufs_delta_db = target_lufs_db - mix_lufs_db
                    pred_mix = pred_mix * 10 ** (lufs_delta_db / 20)

                    mix_filepath = output_dir / f"step{c_idx}-{method_name}-ref={song_section}.wav"
                    torchaudio.save(mix_filepath, pred_mix.view(chs, -1), 44100)


if __name__ == "__main__":
    main()