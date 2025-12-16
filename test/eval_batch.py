import torch
import torchaudio
import yaml
import random
import argparse
import pathlib
import json
import numpy as np
import os
import sys
import glob
import csv
from tqdm import tqdm
import pyloudnorm as pyln

# Add parent directory to path to import mst
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mst.utils import load_diffmst, run_diffmst
import eval_metric

# CLAP import removed to match eval_loop.py behavior


def parse_args():
    parser = argparse.ArgumentParser(description='Batch evaluation of mixing with text prompts')
    
    # Model configs
    parser.add_argument("--config", type=str, required=True, help='Path to model config (e.g. configs/models/naive.yaml)')
    parser.add_argument("--checkpoint", type=str, required=True, help='Path to model checkpoint')
    
    # Dataset configs
    parser.add_argument("--dataset_yaml", type=str, default="data/medley.yaml", help='Path to dataset metadata yaml')
    parser.add_argument("--dataset_root", type=str, required=True, help='Root directory of the dataset (containing the song folders)')
    
    # Evaluation parameters
    parser.add_argument("--num_songs", type=int, default=10, help='Number of songs to evaluate')
    parser.add_argument("--text_prompt", type=str, default="Bright", help='Text prompt to apply')
    parser.add_argument("--target_track_idx", type=int, default=1, help='Track index to apply text prompt to (0-based)')
    parser.add_argument("--output_dir", type=str, default="./eval_batch_outputs", help='Directory to save outputs')
    parser.add_argument("--exp_name", type=str, default="batch_test", help='Experiment name')
    parser.add_argument("--target_lufs", type=float, default=-22.0, help='Target output LUFS')
    
    return parser.parse_args()



def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (torch.Tensor, np.ndarray)):
        if hasattr(obj, 'numel') and obj.numel() == 1:
            return obj.item()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            return str(obj)
    elif isinstance(obj, (float, int, str)):
        return obj
    else:
        return str(obj)

def main():
    args = parse_args()
    
    output_dir = pathlib.Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset metadata
    with open(args.dataset_yaml, 'r') as f:
        dataset_meta = yaml.safe_load(f)
    
    # Get test/val songs
    songs = []
    if 'test' in dataset_meta:
        songs.extend(list(dataset_meta['test'].keys()))
    if 'val' in dataset_meta:
        songs.extend(list(dataset_meta['val'].keys()))
        
    # Filter songs that exist
    valid_songs = []
    for song_path in songs:
        full_path = os.path.join(args.dataset_root, song_path)
        print(f"Checking {full_path}...")
        # Check if directory exists (song_path usually points to RAW folder)
        # But dataset_root might be the parent of MedleyDB_V1 etc.
        # Let's try to construct the path
        # song_path in yaml is like "MedleyDB_V1/V1/Artist_Song/Artist_Song_RAW"
        
        # If dataset_root is "C:/Data", then full path is "C:/Data/MedleyDB_V1/..."
        if os.path.exists(full_path):
            valid_songs.append(song_path)
        else:
            # Try checking if dataset_root is already inside MedleyDB
            # This part depends on how user mounts the dataset. 
            # For now, assume dataset_root + song_path works.
            pass
            
    print(f"Found {len(valid_songs)} valid songs out of {len(songs)} in metadata.")
    
    if len(valid_songs) == 0:
        print("No valid songs found. Please check --dataset_root.")
        return

    # Randomly select songs
    selected_songs = random.sample(valid_songs, min(args.num_songs, len(valid_songs)))
    
    # Load model
    print("Loading model...")
    model, mix_console = load_diffmst(args.config, args.checkpoint)
    model = model.to("cpu")
    mix_console = mix_console.to("cpu")
    
    meter = pyln.Meter(44100)
    
    all_metrics = []
    
    for song_idx, song_rel_path in enumerate(tqdm(selected_songs)):
        song_name = os.path.basename(song_rel_path).replace("_RAW", "")
        print(f"\nProcessing {song_name} ({song_idx+1}/{len(selected_songs)})")
        
        song_dir = os.path.join(args.dataset_root, song_rel_path)
        
        # Find tracks
        track_filepaths = glob.glob(os.path.join(song_dir, "*.wav"))
        if len(track_filepaths) < 2:
            print("Not enough tracks, skipping.")
            continue
            
        # Find mix file
        # Usually mix is in the parent folder of RAW
        parent_dir = os.path.dirname(song_dir)
        mix_files = glob.glob(os.path.join(parent_dir, "*_MIX.wav"))
        if not mix_files:
            print("Mix file not found, skipping.")
            continue
        mix_filepath = mix_files[0]
        
        # Load tracks
        tracks = []
        lengths = []
        for track_filepath in track_filepaths:
            audio, sr = torchaudio.load(track_filepath, backend="soundfile")
            if sr != 44100:
                audio = torchaudio.functional.resample(audio, sr, 44100)
            if audio.shape[0] == 2:
                audio = audio.mean(dim=0, keepdim=True)
            
            chs, seq_len = audio.shape
            for ch_idx in range(chs):
                tracks.append(audio[ch_idx : ch_idx + 1, :])
                lengths.append(audio.shape[-1])
            
        max_length = max(lengths)
        # Pad tracks
        for track_idx in range(len(tracks)):
            tracks[track_idx] = torch.nn.functional.pad(
                tracks[track_idx], (0, max_length - lengths[track_idx])
            )
        
        tracks_tensor = torch.cat(tracks, dim=0)
        tracks_tensor = tracks_tensor.view(1, -1, max_length) # (1, num_tracks, len)
        
        # Load reference mix
        ref_audio, ref_sr = torchaudio.load(mix_filepath, backend="soundfile")
        if ref_sr != 44100:
            ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 44100)
        ref_audio = ref_audio.view(1, 2, -1) # (1, 2, len)
        
        # Ensure lengths match for processing (crop to min length or pad)
        # run_diffmst crops to analysis_len (10s) by default if not specified?
        # It crops to analysis_len inside.
        
        # We want to process a segment. Let's pick a random segment or the beginning.
        # eval_loop uses verse/chorus indices. Here we might just use a fixed segment or random.
        # Let's use a segment from the middle to avoid silence.
        start_idx = 0
        if max_length > 44100 * 30:
            start_idx = 44100 * 10 # Start 10s in
            
        # Slice tracks to 20s (same as eval_loop) to avoid OOM
        slice_len = 44100 * 20
        tracks_slice = tracks_tensor[..., start_idx : start_idx + slice_len].clone()
        
        # --- Step 1: Baseline (Audio Reference Only) ---
        with torch.no_grad():
            res_baseline = run_diffmst(
                tracks_slice,
                ref_audio.clone(),
                model,
                mix_console,
                text=None,
                track_start_idx=0,
                ref_start_idx=start_idx
            )
            (pred_mix_base, pred_tracks_base, pred_track_params, pred_fx_params, pred_master_params) = res_baseline
            
        # --- Step 2: Text Prompt on Target Track ---
        # Target track index
        target_idx = args.target_track_idx
        if target_idx >= tracks_tensor.shape[1]:
            print(f"Target track index {target_idx} out of bounds (num_tracks={tracks_tensor.shape[1]}). Using 0.")
            target_idx = 0
            
        text_input = (target_idx, 1.0, args.text_prompt)
        
        # Prepare reference for text prompt (use output of baseline)
        # If targeting a track, we need separated tracks as reference
        num_tracks = pred_tracks_base.shape[2]
        # pred_tracks_base is (bs, 2, num_tracks, len) -> view as (bs, 2*num_tracks, len)
        ref_audio_text = pred_tracks_base.view(1, 2*num_tracks, -1)
        
        with torch.no_grad():
            res_text = run_diffmst(
                tracks_slice,
                ref_audio_text.clone(),
                model,
                mix_console,
                text=text_input,
                track_start_idx=0,
                ref_start_idx=0,
                prev_track_param_dict=pred_track_params,
                prev_fx_bus_param_dict=pred_fx_params,
                prev_master_bus_param_dict=pred_master_params
            )
            (pred_mix_text, pred_tracks_text, _, _, _) = res_text

        # --- Save Audio ---
        song_out_dir = output_dir / song_name
        song_out_dir.mkdir(exist_ok=True)
        
        # Save Mixes
        torchaudio.save(song_out_dir / "mix_baseline.wav", pred_mix_base.squeeze(0), 44100)
        torchaudio.save(song_out_dir / "mix_text.wav", pred_mix_text.squeeze(0), 44100)
        
        # Save Stems (Target and Sum of Others)
        # Baseline
        # pred_tracks_base shape: (bs, 2, num_tracks, seq_len)
        target_stem_base = pred_tracks_base[0, :, target_idx, :] # (2, len)
        other_stems_base = pred_tracks_base[0].clone()
        other_stems_base[:, target_idx, :] = 0
        sum_others_base = other_stems_base.sum(dim=1) # (2, len)
        
        torchaudio.save(song_out_dir / "target_baseline.wav", target_stem_base, 44100)
        torchaudio.save(song_out_dir / "others_baseline.wav", sum_others_base, 44100)
        
        # Text
        target_stem_text = pred_tracks_text[0, :, target_idx, :]
        other_stems_text = pred_tracks_text[0].clone()
        other_stems_text[:, target_idx, :] = 0
        sum_others_text = other_stems_text.sum(dim=1)
        
        torchaudio.save(song_out_dir / "target_text.wav", target_stem_text, 44100)
        torchaudio.save(song_out_dir / "others_text.wav", sum_others_text, 44100)
        
        # --- Compute Metrics ---
        # Audio Metrics (Spectral Centroid, Band Ratio, Crest Factor)
        # We compare Target Track (Base vs Text) and Others (Base vs Text)
        
        def compute_audio_metrics(waveform, name_suffix):
            # waveform: (2, len) -> mix to mono for metrics
            mono = waveform.mean(dim=0).numpy()
            return {
                f"spectral_centroid_{name_suffix}": eval_metric.get_spectral_centroid(mono),
                f"band_ratio_{name_suffix}": eval_metric.get_band_ratio(mono),
                f"crest_factor_{name_suffix}": eval_metric.get_crest_factor(mono)
            }

        metrics = {}
        metrics.update(compute_audio_metrics(target_stem_base, "target_audio_base"))
        metrics.update(compute_audio_metrics(target_stem_text, "target_text_modified"))
        metrics.update(compute_audio_metrics(sum_others_base, "others_audio_base"))
        metrics.update(compute_audio_metrics(sum_others_text, "others_text_modified"))
        

            
        metrics["song"] = song_name
        all_metrics.append(metrics)
        
        # Save Parameters
        params = {
            "track_params": make_serializable(pred_track_params),
            "fx_bus_params": make_serializable(pred_fx_params),
            "master_bus_params": make_serializable(pred_master_params)
        }
        with open(song_out_dir / "parameters.json", 'w') as f:
            json.dump(params, f, indent=4)

    # --- Aggregate Results ---
    if not all_metrics:
        print("No metrics computed.")
        return

    # Save to CSV
    keys = all_metrics[0].keys()
    with open(output_dir / "metrics.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_metrics)
        
    # Compute Averages
    avg_metrics = {}
    for k in keys:
        if k == "song": continue
        values = [m[k] for m in all_metrics]
        avg_metrics[k] = sum(values) / len(values)
        
    print("\nAverage Metrics:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")
        
    with open(output_dir / "avg_metrics.json", 'w') as f:
        json.dump(avg_metrics, f, indent=4)

if __name__ == "__main__":
    main()
