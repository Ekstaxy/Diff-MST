# run pretrained models over evaluation set to generate audio examples for the listening test
import os
import argparse
import torch
import torchaudio
import pyloudnorm as pyln
from mst.utils import load_diffmst, run_diffmst


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


def parse_args():
    parser = argparse.ArgumentParser(description='Generate audio examples for listening test')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Computation device')

    # Model paths
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')

    # Audio paths
    parser.add_argument('--tracks', type=str, required=True,
                        help='Path to tracks folder')
    parser.add_argument('--reference', type=str, required=True,
                        help='Path to reference mix WAV file')

    # Optional settings
    parser.add_argument('--output', type=str, default='outputs/listen',
                        help='Output directory')
    parser.add_argument('--name', type=str, default='experiment',
                        help='Experiment name')
    parser.add_argument('--target-lufs', type=float, default=-22.0,
                        help='Target output LUFS')

    # Verse/Chorus indices
    parser.add_argument('--track-verse-idx', type=int, required=True,
                        help='Track verse start index (samples)')
    parser.add_argument('--track-chorus-idx', type=int, required=True,
                        help='Track chorus start index (samples)')
    parser.add_argument('--ref-verse-idx', type=int, required=True,
                        help='Reference verse start index (samples)')
    parser.add_argument('--ref-chorus-idx', type=int, required=True,
                        help='Reference chorus start index (samples)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    meter = pyln.Meter(44100)
    target_lufs_db = args.target_lufs
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model to {args.device}...")
    model, mix_console = load_diffmst(
        args.config,
        args.checkpoint,
        map_location=args.device,
    )

    # Move to device
    model = model.to(args.device)
    mix_console = mix_console.to(args.device)
    model.eval()
    mix_console.eval()

    print(f"✓ Model loaded on {args.device}")

    methods = {
        "diffmst-16": {
            "model": (model, mix_console),
            "func": run_diffmst,
        },
        "sum": {
            "model": (None, None),
            "func": equal_loudness_mix,
        },
    }

    # Single example from arguments
    example = {
        "tracks": args.tracks,
        "track_verse_start_idx": args.track_verse_idx,
        "track_chorus_start_idx": args.track_chorus_idx,
        "ref": args.reference,
        "ref_verse_start_idx": args.ref_verse_idx,
        "ref_chorus_start_idx": args.ref_chorus_idx,
    }

    example_name = args.name
    print(example_name)
    example_dir = os.path.join(output_dir, example_name)
    os.makedirs(example_dir, exist_ok=True)

    # load reference mix
    ref_audio, ref_sr = torchaudio.load(example["ref"], backend="soundfile")
    if ref_sr != 44100:
        ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 44100)
    print(ref_audio.shape, ref_sr)

    # first find all the tracks
    track_filepaths = []
    for root, dirs, files in os.walk(example["tracks"]):
        for filepath in files:
            if filepath.endswith(".wav"):
                track_filepaths.append(os.path.join(root, filepath))

    print(f"Found {len(track_filepaths)} tracks.")

    # load the tracks
    tracks = []
    lengths = []
    for track_idx, track_filepath in enumerate(track_filepaths):
        audio, sr = torchaudio.load(track_filepath, backend="soundfile")

        if sr != 44100:
            audio = torchaudio.functional.resample(audio, sr, 44100)

        # loudness normalize the tracks to -48 LUFS
        lufs_db = meter.integrated_loudness(audio.permute(1, 0).numpy())
        # lufs_delta_db = -48 - lufs_db
        # audio = audio * 10 ** (lufs_delta_db / 20)

        #print(track_idx, os.path.basename(track_filepath), audio.shape, sr, lufs_db)

        if audio.shape[0] == 2:
            audio = audio.mean(dim=0, keepdim=True)

        chs, seq_len = audio.shape

        for ch_idx in range(chs):
            tracks.append(audio[ch_idx : ch_idx + 1, :])
            lengths.append(audio.shape[-1])
    print("Loaded tracks.")

    # find max length and pad if shorter
    max_length = max(lengths)
    for track_idx in range(len(tracks)):
        tracks[track_idx] = torch.nn.functional.pad(
            tracks[track_idx], (0, max_length - lengths[track_idx])
        )
    print("Padded tracks.")

    # stack into a tensor
    tracks = torch.cat(tracks, dim=0)
    tracks = tracks.view(1, -1, max_length)
    ref_audio = ref_audio.view(1, 2, -1)

    # crop tracks to max of 60 seconds or so
    # tracks = tracks[..., :4194304]

    print(tracks.shape)

    # create a sum mix with equal loudness
    sum_mix = torch.sum(tracks, dim=1, keepdim=True).squeeze(0)
    sum_filepath = os.path.join(example_dir, f"{example_name}-sum.wav")

    # loudness normalize the sum mix
    sum_lufs_db = meter.integrated_loudness(sum_mix.permute(1, 0).numpy())
    lufs_delta_db = target_lufs_db - sum_lufs_db
    sum_mix = sum_mix * 10 ** (lufs_delta_db / 20)

    torchaudio.save(sum_filepath, sum_mix.view(1, -1), 44100)
    print("Sum mix saved.")

    # save the reference mix
    ref_filepath = os.path.join(example_dir, "ref-full.wav")
    torchaudio.save(ref_filepath, ref_audio.squeeze(), 44100)
    print("Reference mix saved.")

    for song_section in ["verse", "chorus"]:
        print("Mixing", song_section)
        if song_section == "verse":
            track_start_idx = example["track_verse_start_idx"]
            ref_start_idx = example["ref_verse_start_idx"]
        else:
            track_start_idx = example["track_chorus_start_idx"]
            ref_start_idx = example["ref_chorus_start_idx"]

        if track_start_idx + 262144 > tracks.shape[-1]:
            print("Tracks too short for this section.")
        if ref_start_idx + 262144 > ref_audio.shape[-1]:
            print("Reference too short for this section.")

        # crop the tracks to create a mix twice the size of the reference section
        mix_tracks = tracks
        # [..., track_start_idx : track_start_idx + (262144 * 2)]
        mix_tracks = tracks[..., track_start_idx : track_start_idx + (262144 * 2)]
        track_start_idx = 0
        print("mix_tracks", mix_tracks.shape)

        # save the reference mix section for analysis
        ref_analysis = ref_audio[..., ref_start_idx : ref_start_idx + 262144]

        # create mixes varying the loudness of the reference
        for ref_loudness_target in [-24, -16, -14.0, -12, -6]:
            print("Ref loudness", ref_loudness_target)
            ref_filepath = os.path.join(
                example_dir,
                f"ref-analysis-{song_section}-lufs-{ref_loudness_target:0.0f}.wav",
            )

            # loudness normalize the reference mix section to -14 LUFS
            ref_lufs_db = meter.integrated_loudness(
                ref_analysis.squeeze().permute(1, 0).numpy()
            )
            lufs_delta_db = ref_loudness_target - ref_lufs_db
            ref_analysis = ref_analysis * 10 ** (lufs_delta_db / 20)

            torchaudio.save(ref_filepath, ref_analysis.squeeze(), 44100)

            for method_name, method in methods.items():
                print(method_name)
                # tracks (torch.Tensor): Set of input tracks with shape (bs, num_tracks, seq_len)
                # ref_audio (torch.Tensor): Reference mix with shape (bs, 2, seq_len)

                if method_name == "sum":
                    if ref_loudness_target != -16:
                        continue

                if method_name == "sum" and song_section == "chorus":
                    continue

                model, mix_console = method["model"]
                func = method["func"]

                # Ensure model is on correct device
                model = model.to(args.device) if model is not None else None
                mix_console = mix_console.to(args.device) if mix_console is not None else None

                print(tracks.shape, ref_audio.shape)

                with torch.no_grad():
                    result = func(
                        tracks.clone(),
                        ref_audio.clone(),
                        model,
                        mix_console,
                        track_start_idx=track_start_idx,
                        ref_start_idx=ref_start_idx,
                    )

                    (
                        pred_mix,
                        pred_track_param_dict,
                        pred_fx_bus_param_dict,
                        pred_master_bus_param_dict,
                    ) = result

                bs, chs, seq_len = pred_mix.shape

                # loudness normalize the output mix
                mix_lufs_db = meter.integrated_loudness(
                    pred_mix.squeeze(0).permute(1, 0).numpy()
                )
                print(mix_lufs_db)
                lufs_delta_db = target_lufs_db - mix_lufs_db
                pred_mix = pred_mix * 10 ** (lufs_delta_db / 20)

                # save resulting audio and parameters
                mix_filepath = os.path.join(
                    example_dir,
                    f"{example_name}-{method_name}-ref={song_section}-lufs-{ref_loudness_target:0.0f}.wav",
                )
                torchaudio.save(mix_filepath, pred_mix.view(chs, -1), 44100)

                # also save only the analysis section
                mix_analysis = pred_mix[
                    ..., track_start_idx : track_start_idx + (2 * 262144)
                ]

                # loudness normalize the output mix
                mix_lufs_db = meter.integrated_loudness(
                    mix_analysis.squeeze(0).permute(1, 0).numpy()
                )
                print(mix_lufs_db)
                mix_analysis = mix_analysis * 10 ** (lufs_delta_db / 20)

                mix_filepath = os.path.join(
                    example_dir,
                    f"{example_name}-{method_name}-analysis-{song_section}-lufs-{ref_loudness_target:0.0f}.wav",
                )
                torchaudio.save(mix_filepath, mix_analysis.view(chs, -1), 44100)

    print()
