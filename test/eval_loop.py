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
                        default=["/mnt/gestalt/home/rakec/data/diff-mst/MedleyDB_V2/V2/TleilaxEnsemble_Late/TleilaxEnsemble_Late_MIX.wav", (1, "text")],
                        help="Control information (file paths for audio, text prompts for text)")
    
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

                ref_analysis = ref_audio[..., ref_start_idx : ref_start_idx + 44100 * 10]
                ref_loudness_target = -14.0
                ref_filepath = output_dir / f"ref_{song_section}_lufs{ref_loudness_target}.wav"

                ref_lufs_db = meter.integrated_loudness(
                    ref_analysis.squeeze().permute(1, 0).numpy()
                )
                lufs_delta_db = ref_loudness_target - ref_lufs_db
                ref_analysis = ref_analysis * 10 ** (lufs_delta_db / 20)

                torchaudio.save(ref_filepath, ref_analysis.squeeze(), 44100)

if __name__ == "__main__":
    main()