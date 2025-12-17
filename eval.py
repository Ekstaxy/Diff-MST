import torch
import pathlib
import argparse
import torchaudio
from tqdm import tqdm
import pyloudnorm as pyln
from typing import Optional

from mst.utils import load_diffmst, run_diffmst

class arguments:
    config: str
    checkpoint: str
    tracks_path: str
    control_type: str
    control_info: None
    track_idx: int
    ref_idx: int
    output_dir: str
    exp_name: str
    target_lufs: float
    sum_only: bool  

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

def mix_sum_only(args: arguments):
    """
    Mix audio using sum-only baseline.
    Args:
        args: Argument namespace containing configuration.
    Returns:
        Tuple containing predicted mix, mixed tracks, track parameters,
        FX bus parameters, and master bus parameters.
    """


    meter = pyln.Meter(44100)
    target_lufs_db = args.target_lufs
    output_dir = pathlib.Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

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

    track_start_idx = args.track_chorus_idx

    mix_tracks = tracks[..., track_start_idx : track_start_idx + (44100 * 10)]

    sum_mix, _, _, _ = equal_loudness_mix(mix_tracks)

    mix_lufs_db = meter.integrated_loudness(
        sum_mix.squeeze(0).permute(1, 0).numpy()
    )
    lufs_delta_db = target_lufs_db - mix_lufs_db
    sum_mix = sum_mix * 10 ** (lufs_delta_db / 20)

    mix_filepath = output_dir / f"sum-baseline-lufs-{int(target_lufs_db)}.wav"
    torchaudio.save(mix_filepath, sum_mix.view(2, -1), 44100)
    return sum_mix, None, None, None, None

def mix_audio_ref(args: arguments):
    """
    Mix audio using audio-based reference.
    Args:
        args: Argument namespace containing configuration.
    Returns:
        Tuple containing predicted mix, mixed tracks, track parameters,
        FX bus parameters, and master bus parameters.
    """

    meter = pyln.Meter(44100)
    target_lufs_db = args.target_lufs
    output_dir = pathlib.Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    method = {
            "model": load_diffmst(
                config_path=args.config,
                ckpt_path=args.checkpoint
            ),
            "func": run_diffmst
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
    c_type = args.control_type
    assert c_type in ["audio", "text"], f"Unsupported control type: {c_type}"

    example = {
        "tracks": args.tracks_path,
        "track_start_idx": args.track_idx,
        "ref": args.control_info,
        "ref_start_idx": args.ref_idx,
    }

    ref_audio, ref_sr = torchaudio.load(example["ref"], backend="soundfile")
    if ref_sr != 44100:
        ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 44100)
    ref_audio = ref_audio.view(1, 2, -1)
    print(f"[INFO] reference audio shape: {ref_audio.shape}")

    track_start_idx = example["track_start_idx"]
    ref_start_idx = example["ref_start_idx"]

    if track_start_idx + 44100 * 10 > tracks.shape[-1]:
        print(f"[Warning] Tracks too short for this section.")
    if ref_start_idx + 44100 * 10 > ref_audio.shape[-1]:
        print(f"[Warning] Reference too short for this section.")

    mix_tracks = tracks
    mix_tracks = tracks[..., track_start_idx : track_start_idx + (44100 * 10 * 2)]
    track_start_idx = 0


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
    return pred_mix, pred_mixed_tracks, pred_track_param_dict, pred_fx_bus_param_dict, pred_master_bus_param_dict
                
def mix_text_optimize(
        args: arguments, 
        prev_fx_bus_param_dict: dict, 
        prev_master_bus_param_dict: dict, 
        prev_track_param_dict: dict, 
        pred_mix: Optional[torch.Tensor] = None, 
        pred_mixed_tracks: Optional[torch.Tensor] = None
    ):
    """
    Mix audio using text-based control optimization.
    Args:
        args: Argument namespace containing configuration.
        prev_fx_bus_param_dict: Previous FX bus parameters.
        prev_master_bus_param_dict: Previous master bus parameters.
        prev_track_param_dict: Previous track parameters.
        pred_mix: Previously predicted mix audio (optional).
        pred_mixed_tracks: Previously predicted mixed tracks (optional).
    Returns:
        Tuple containing predicted mix, mixed tracks, track parameters,
        FX bus parameters, and master bus parameters.
    """

    meter = pyln.Meter(44100)
    target_lufs_db = args.target_lufs
    output_dir = pathlib.Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = {
        "model": load_diffmst(
            config_path=args.config,
            ckpt_path=args.checkpoint
        ),
        "func": run_diffmst
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

    text = args.control_info
    if not(isinstance(text, tuple) or isinstance(text, list)):
        raise ValueError("For text control, control_info must be a tuple or list.")
    
    print(f"[INFO] Using text prompt: {text[2]}, weight: {text[1]}, track: {text[0]}")
    
    example = {
        "tracks": args.tracks_path,
        "track_start_idx": args.track_idx,
        "ref": args.control_info,
        "ref_start_idx": args.ref_idx,
    }

    num_tracks = pred_mixed_tracks.shape[2]     # pred_mixed_tracks: (bs, 2, num_tracks, seq_len)
    if example["ref"][0] < -1 or example["ref"][0] >= num_tracks:
        raise ValueError(f"Invalid track index {example['ref'][0]} for {num_tracks} tracks.")

    if example["ref"][0] == -1:
        if pred_mix is None:
            raise ValueError("pred_mix must be provided when reference is mix.")
        ref_audio = pred_mix
    else:
        if pred_mixed_tracks is None:
            raise ValueError("pred_mixed_tracks must be provided when reference is track.")
        ref_audio = pred_mixed_tracks
        ref_audio = ref_audio.view(1, 2*num_tracks, -1)

    print(f"[INFO] reference audio shape: {ref_audio.shape}")
    
    prev_fx_bus_param_dict = pred_fx_bus_param_dict
    prev_track_param_dict = pred_track_param_dict
    prev_master_bus_param_dict = pred_master_bus_param_dict

    track_start_idx = example["track_start_idx"]
    ref_start_idx = example["ref_start_idx"]

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

    return pred_mix, pred_mixed_tracks, pred_track_param_dict, pred_fx_bus_param_dict, pred_master_bus_param_dict

if __name__ == "__main__":
    arg1 = arguments(
        config = "./test/naive.yaml".
        checkpoint = "./epoch=247-step=77624.ckpt",
        tracks_path = "./path_to_tracks",
        control_type = "audio",
        control_info = "./path_to_reference.wav",
        track_idx = 441000,
        ref_idx = 441000,
        output_dir = "./output",
        exp_name = "experiment1",
        target_lufs = -22.0,
        sum_only = False  
    )

    arg2 = arguments(
        config = "./test/naive.yaml".
        checkpoint = "./epoch=247-step=77624.ckpt",
        tracks_path = "./path_to_tracks",
        control_type = "text",
        control_info = (0, 1.0, "Make the vocals louder and clearer."),
        track_idx = 441000,
        ref_idx = 0,
        output_dir = "./output",
        exp_name = "experiment1",
        target_lufs = -22.0,
        sum_only = False  
    )

    result_sum = mix_sum_only(arg1)
    result_audio_ref = mix_audio_ref(arg1)
    (pred_mix, 
     pred_mixed_tracks, 
     pred_track_param_dict, 
     pred_fx_bus_param_dict, 
     pred_master_bus_param_dict) = result_audio_ref
    result_text_opt = mix_text_optimize(arg2, 
                                        prev_fx_bus_param_dict=pred_fx_bus_param_dict, 
                                        prev_master_bus_param_dict=pred_master_bus_param_dict, 
                                        prev_track_param_dict=pred_track_param_dict, 
                                        pred_mix=pred_mix, 
                                        pred_mixed_tracks=pred_mixed_tracks)
    

