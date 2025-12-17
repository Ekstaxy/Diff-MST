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
from mst.loss import AudioFeatureLoss
import eval_metric

def compute_clap_similarity(clap_model, audio, text, original_sr=44100):
    # audio: (len,) or (channels, len)
    # text: str
    
    # Ensure tensor
    if not torch.is_tensor(audio):
        audio = torch.tensor(audio)
        
    # Mix to mono if stereo
    if audio.dim() == 2:
        audio = audio.mean(dim=0)
        
    # Resample to 48k
    if original_sr != 48000:
        audio = torchaudio.functional.resample(audio, original_sr, 48000)
        
    # CLAP expects (batch, len)
    audio = audio.unsqueeze(0) # (1, len)
    
    # Get embeddings
    with torch.no_grad():
        # Ensure model is on same device as audio (or vice versa)
        # In this script, everything is CPU
        audio_embed = clap_model.get_audio_embedding_from_data(x=audio, use_tensor=True)
        text_embed = clap_model.get_text_embedding([text], use_tensor=True)
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(audio_embed, text_embed)
    return similarity.item()

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
    parser.add_argument("--num_songs", type=int, default=1, help='Number of songs to evaluate')
    parser.add_argument("--seed", type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument("--text_prompt", type=str, default="Bright", help='Text prompt to apply')
    parser.add_argument("--interpolation", type=str, default="linear", help='Interpolation method: linear or slerp')
    parser.add_argument("--target_track_idx", type=int, default=1, help='Track index to apply text prompt to (0-based)')
    parser.add_argument("--output_dir", type=str, default="./eval_batch_outputs", help='Directory to save outputs')
    parser.add_argument("--exp_name", type=str, default="batch_test", help='Experiment name')
    parser.add_argument("--target_lufs", type=float, default=-22.0, help='Target output LUFS')
    parser.add_argument("--num_iterations", type=int, default=1, help='Number of text prompt iterations')
    
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

def normalize_audio(mix, stems, target_lufs, meter, name="mix"):
    # mix: (1, 2, len)
    # stems: (1, 2, num_tracks, len)
    try:
        # Check for silence first
        if mix.abs().max() < 1e-6:
            print(f"Warning: {name} is silent (max < 1e-6). Skipping normalization.")
            return mix, stems

        mix_np = mix.squeeze(0).permute(1, 0).cpu().numpy() # (len, 2)
        mix_lufs_db = meter.integrated_loudness(mix_np)
        
        if mix_lufs_db == -float('inf'):
                print(f"Warning: {name} LUFS is -inf. Applying default gain +26dB.")
                gain_db = 26.0
        else:
            lufs_delta_db = target_lufs - mix_lufs_db
            gain_db = lufs_delta_db
        
        print(f"Normalizing {name}: Current LUFS = {mix_lufs_db:.2f}, Gain = {gain_db:.2f} dB")
        
        mix = mix * 10 ** (gain_db / 20)
        stems = stems * 10 ** (gain_db / 20)
        return mix, stems
    except Exception as e:
        print(f"Warning: Could not normalize {name}: {e}. Applying default gain +26dB.")
        # Fallback: Input is likely around -48 LUFS, target is -22 LUFS -> +26dB
        gain_db = 26.0
        mix = mix * 10 ** (gain_db / 20)
        stems = stems * 10 ** (gain_db / 20)
        return mix, stems

def normalize_stem(waveform, target_lufs, meter, name="stem"):
    # waveform: (2, len)
    try:
        if waveform.abs().max() < 1e-6:
            print(f"Warning: {name} is silent.")
            return waveform
        
        wav_np = waveform.permute(1, 0).cpu().numpy()
        lufs = meter.integrated_loudness(wav_np)
        
        if lufs == -float('inf'):
            gain_db = 26.0
        else:
            gain_db = target_lufs - lufs
        
        return waveform * 10 ** (gain_db / 20)
    except Exception as e:
        print(f"Warning: Could not normalize {name}: {e}")
        return waveform * 10 ** (26.0 / 20) # Fallback gain

def compute_audio_metrics(waveform, name_suffix):
    # waveform: (2, len) -> mix to mono for metrics
    # mono = waveform.mean(dim=0).numpy()
    return {
        f"loudness_{name_suffix}": eval_metric.get_loudness(waveform.numpy()),
        f"panning_{name_suffix}": eval_metric.get_panning(waveform.numpy()),
        f"mid_side_ratio_{name_suffix}": eval_metric.get_mid_side_ratio(waveform.numpy()),
        f"spectral_centroid_{name_suffix}": eval_metric.get_spectral_centroid(waveform.mean(dim=0).numpy()),
        f"band_ratio_{name_suffix}": eval_metric.get_band_ratio(waveform.mean(dim=0).numpy()),
        f"crest_factor_{name_suffix}": eval_metric.get_crest_factor(waveform.mean(dim=0).numpy())
    }

def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
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
    # selected_songs = random.sample(valid_songs, min(args.num_songs, len(valid_songs)))
    selected_songs = valid_songs[:args.num_songs]
    
    # Load model
    print("Loading model...")
    model, mix_console = load_diffmst(args.config, args.checkpoint)
    model = model.to("cpu")
    mix_console = mix_console.to("cpu")
    
    meter = pyln.Meter(44100)
    
    # Initialize AudioFeatureLoss
    af_loss_fn = AudioFeatureLoss([0.1, 0.001, 1.0, 1.0, 0.1], 44100)

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
        
        # Determine target track index early to find active slice
        target_idx = args.target_track_idx
        if target_idx >= tracks_tensor.shape[1]:
            print(f"Target track index {target_idx} out of bounds. Using 0.")
            target_idx = 0

        # We want to process a segment. Let's pick a random segment or the beginning.
        # eval_loop uses verse/chorus indices. Here we might just use a fixed segment or random.
        # Let's use a segment from the middle to avoid silence.
        start_idx = 0
        slice_len = 44100 * 10
        
        # Search for a slice where the target track is active
        found_active = False
        if max_length > slice_len:
            # Scan in 5s increments
            step = 44100 * 5
            best_energy = -1.0
            best_idx = 0
            
            # Limit scan to avoid taking too long on very long tracks
            scan_end = max_length - slice_len
            
            for try_idx in range(0, scan_end, step):
                # Check energy of target track in this slice
                # tracks_tensor: (1, num_tracks, len)
                target_slice = tracks_tensor[0, target_idx, try_idx : try_idx + slice_len]
                energy = target_slice.pow(2).mean().item()
                
                if energy > 1e-4: # Threshold for "active"
                    start_idx = try_idx
                    found_active = True
                    break
                
                if energy > best_energy:
                    best_energy = energy
                    best_idx = try_idx
            
            if not found_active:
                print(f"Warning: Could not find active slice for track {target_idx}. Using slice with max energy.")
                start_idx = best_idx
        
        # Slice tracks to 10s (same as eval_loop) to avoid OOM
        if start_idx + slice_len > tracks_tensor.shape[-1]:
            start_idx = 0
            if tracks_tensor.shape[-1] < slice_len:
                slice_len = tracks_tensor.shape[-1]
                
        tracks_slice = tracks_tensor[..., start_idx : start_idx + slice_len].clone()
        ref_slice = ref_audio[..., start_idx : start_idx + slice_len].clone()

        # --- Select top 8 active tracks ---
        # Calculate energy of each track in the slice
        track_energies = tracks_slice.squeeze(0).pow(2).mean(dim=-1) # (num_tracks,)
        
        # We must include target_idx
        selected_indices = [target_idx]
        
        # Get indices sorted by energy
        sorted_indices = torch.argsort(track_energies, descending=True)
        
        for idx in sorted_indices:
            idx = idx.item()
            if len(selected_indices) >= 8:
                break
            if idx not in selected_indices:
                selected_indices.append(idx)
        
        selected_indices.sort()
        
        # Update tracks_slice
        tracks_slice = tracks_slice[:, selected_indices, :]
        
        # Update target_idx to new index
        target_idx = selected_indices.index(target_idx)
        print(f"Selected {len(selected_indices)} tracks. New target index: {target_idx}")
        
        # --- Step 1: Baseline (Audio Reference Only) ---
        with torch.no_grad():
            res_baseline = run_diffmst(
                tracks_slice,
                ref_slice,
                model,
                mix_console,
                text=None,
                interpolation=args.interpolation,
                track_start_idx=0,
                ref_start_idx=0,
                use_master_bus=True
            )
            (pred_mix_base, pred_tracks_base, pred_track_params, pred_fx_params, pred_master_params) = res_baseline
            
        # --- Step 2: Text Prompt on Target Track ---
        # Target track index determined earlier

        # track_idx, text_alpha, style_alpha, text_prompt, is_panning = text
        
        if args.num_iterations > 0:
            text_input = (target_idx, 1.0, 1.0, args.text_prompt, False)
            
            # Initial reference and params from baseline
            current_ref_tracks = pred_tracks_base # (bs, 2, num_tracks, len)
            current_track_params = pred_track_params
            current_fx_params = pred_fx_params
            current_master_params = pred_master_params
            
            pred_mix_text = None
            pred_tracks_text = None

            for i in range(args.num_iterations):
                # Prepare reference for text prompt
                # If targeting a track, we need separated tracks as reference
                num_tracks = current_ref_tracks.shape[2]
                # current_ref_tracks is (bs, 2, num_tracks, len) -> view as (bs, 2*num_tracks, len)
                ref_audio_text = current_ref_tracks.view(1, 2*num_tracks, -1)
                
                with torch.no_grad():
                    res_text = run_diffmst(
                        tracks_slice,
                        ref_audio_text.clone(),
                        model,
                        mix_console,
                        text=text_input,
                        interpolation=args.interpolation,
                        track_start_idx=0,
                        ref_start_idx=0,
                        prev_track_param_dict=current_track_params,
                        prev_fx_bus_param_dict=current_fx_params,
                        prev_master_bus_param_dict=current_master_params,
                        use_master_bus=True
                    )
                    (pred_mix_text, pred_tracks_text, current_track_params, current_fx_params, current_master_params) = res_text
                
                # Update reference for next iteration
                current_ref_tracks = pred_tracks_text

            # Update params to the final ones for saving
            pred_track_params = current_track_params
            pred_fx_params = current_fx_params
            pred_master_params = current_master_params
        else:
            pred_mix_text = None
            pred_tracks_text = None

        # --- Sum Mix ---
        sum_mix = tracks_slice.sum(dim=1, keepdim=True).repeat(1, 2, 1) # (1, 2, len)

        # Baseline
        target_stem_base = pred_tracks_base[0, :, target_idx, :] # (2, len)
        other_stems_base = pred_tracks_base[0].clone()
        other_stems_base[:, target_idx, :] = 0
        sum_others_base = other_stems_base.sum(dim=1) # (2, len)
        
        if args.num_iterations > 0:
            # Text
            target_stem_text = pred_tracks_text[0, :, target_idx, :]
            other_stems_text = pred_tracks_text[0].clone()
            other_stems_text[:, target_idx, :] = 0
            sum_others_text = other_stems_text.sum(dim=1)

        # --- Compute Metrics ---
        # Audio Metrics (Spectral Centroid, Band Ratio, Crest Factor)
        # We compare Target Track (Base vs Text) and Others (Base vs Text)

        metrics = {}
        metrics.update(compute_audio_metrics(target_stem_base, "target_audio_base"))
        metrics.update(compute_audio_metrics(sum_others_base, "others_audio_base"))

        if args.num_iterations > 0:
            metrics.update(compute_audio_metrics(target_stem_text, "target_text_modified"))
            metrics.update(compute_audio_metrics(sum_others_text, "others_text_modified"))

        # Compute AF Loss
        # pred_mix_base: (1, 2, len)
        # ref_slice: (1, 2, len)
        af_losses_base = af_loss_fn(pred_mix_base, ref_slice)
        if args.num_iterations > 0:
            af_losses_text = af_loss_fn(pred_mix_text, ref_slice)
        af_losses_sum = af_loss_fn(sum_mix, ref_slice)

        # Calculate and log Total AF Loss (sum of all components)
        metrics["AF_base_total"] = sum(af_losses_base.values()).item()
        for k, v in af_losses_base.items():
            metrics[f"AF_base_{k}"] = v.item()

        if args.num_iterations > 0:
            metrics["AF_text_total"] = sum(af_losses_text.values()).item()
            for k, v in af_losses_text.items():
                metrics[f"AF_text_{k}"] = v.item()

            # CLAP Similarity
            # Try to access CLAP model from the loaded model's text_encoder
            if hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'model'):
                clap_model = model.text_encoder.model
                
                origin_target = tracks_slice[0, target_idx, :]
                pred_target = target_stem_text
                
                metrics["CLAP_text_target_origin"] = compute_clap_similarity(clap_model, origin_target, args.text_prompt)
                metrics["CLAP_text_target_pred"] = compute_clap_similarity(clap_model, pred_target, args.text_prompt)
            else:
                print("Warning: Could not find CLAP model in text_encoder. Skipping CLAP metrics.")

        metrics["AF_sum_total"] = sum(af_losses_sum.values()).item()
        for k, v in af_losses_sum.items():
            metrics[f"AF_sum_{k}"] = v.item()

        # --- Loudness Normalization ---
        # Normalize mixes to target LUFS
        pred_mix_base, pred_tracks_base = normalize_audio(pred_mix_base, pred_tracks_base, args.target_lufs, meter, "baseline")
        if args.num_iterations > 0:
            pred_mix_text, pred_tracks_text = normalize_audio(pred_mix_text, pred_tracks_text, args.target_lufs, meter, "text")
        
        dummy_stems_sum = tracks_slice.unsqueeze(1).repeat(1, 2, 1, 1)
        sum_mix, _ = normalize_audio(sum_mix, dummy_stems_sum, args.target_lufs, meter, "sum")

        # --- Save Audio ---
        song_out_dir = output_dir / song_name
        song_out_dir.mkdir(exist_ok=True)
        
        # Save Mixes
        torchaudio.save(song_out_dir / "mix_baseline.wav", pred_mix_base.squeeze(0), 44100)
        if args.num_iterations > 0:
            torchaudio.save(song_out_dir / "mix_text.wav", pred_mix_text.squeeze(0), 44100)
        torchaudio.save(song_out_dir / "mix_sum.wav", sum_mix.squeeze(0), 44100)
        
        # Normalize stems for audibility (Note: this changes relative mix balance in the saved file)
        target_stem_base = normalize_stem(target_stem_base, args.target_lufs, meter, "target_base")
        sum_others_base = normalize_stem(sum_others_base, args.target_lufs, meter, "others_base")

        torchaudio.save(song_out_dir / "target_baseline.wav", target_stem_base, 44100)
        torchaudio.save(song_out_dir / "others_baseline.wav", sum_others_base, 44100)
        
        if args.num_iterations > 0:
            target_stem_text = normalize_stem(target_stem_text, args.target_lufs, meter, "target_text")
            sum_others_text = normalize_stem(sum_others_text, args.target_lufs, meter, "others_text")

            torchaudio.save(song_out_dir / "target_text.wav", target_stem_text, 44100)
            torchaudio.save(song_out_dir / "others_text.wav", sum_others_text, 44100)
            
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
        
    print("\n" + "="*60)
    print(f"{'Metric':<40} | {'Value':<15}")
    print("-" * 60)
    for k, v in avg_metrics.items():
        print(f"{k:<40} | {v:.4f}")
    print("="*60 + "\n")
        
    with open(output_dir / "avg_metrics.json", 'w') as f:
        json.dump(avg_metrics, f, indent=4)

if __name__ == "__main__":
    main()
