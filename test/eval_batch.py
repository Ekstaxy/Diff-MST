import math
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

from mst.utils import load_diffmst, run_diffmst, batch_stereo_peak_normalize, batch_stereo_tracks_peak_normalize
from mst.loss import AudioFeatureLoss, CLAPFeatureLoss
from utils import normalize_audio, normalize_stem
import eval_metric
import matplotlib.pyplot as plt

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
        
        # Normalize embeddings
        audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
        text_embed = torch.nn.functional.normalize(text_embed, dim=-1)
        
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
    parser.add_argument("--text_prompt", type=str, nargs='+', default=["Bright"], help='Text prompt to apply')
    parser.add_argument("--neg_prompt", type=str, nargs='+', default=None, help='Negative text prompt to apply')
    parser.add_argument("--custom_reference", type=str, default=None, help='Path to custom reference audio (overrides dataset mix)')
    parser.add_argument("--interpolation", type=str, default="linear", help='Interpolation method: linear or slerp')
    parser.add_argument("--target_track_idx", type=int, default=-1, help='Track index to apply text prompt to (0-based)')
    parser.add_argument("--output_dir", type=str, default="./eval_batch_outputs", help='Directory to save outputs')
    parser.add_argument("--exp_name", type=str, default="batch_test", help='Experiment name')
    parser.add_argument("--target_lufs", type=float, default=-22.0, help='Target output LUFS')
    parser.add_argument("--num_iterations", type=int, default=1, help='Number of text prompt iterations (Deprecated, used for text interpolation)')
    parser.add_argument("--style_alpha", type=float, default=0.5, help='Style interpolation alpha for text prompt')
    parser.add_argument("--text_alpha", type=float, default=1.0, help='Text interpolation alpha for text prompt')
    parser.add_argument("--is_panning", type=bool, default=False, help='Whether the text prompt is for panning or not')
    
    # ITO Parameters
    parser.add_argument("--ito_num_step", type=int, default=50, help='Number of ITO steps')
    parser.add_argument("--ito_lr", type=float, default=2e-4, help='Learning rate for ITO optimization')
    parser.add_argument("--clap_checkpoint", type=str, default=None, help='Path to CLAP model checkpoint')
    parser.add_argument("--prior_loss_weight", type=float, default=0.01, help='Weight for prior loss in ITO')
    parser.add_argument("--prior_stats_path", type=str, default="test/track_prior_stats.json", help='Path to prior stats JSON for ITO')
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

def compute_audio_metrics(waveform, name_suffix):
    # waveform: (2, len) -> mix to mono for metrics
    # mono = waveform.mean(dim=0).numpy()
    metrics = {
        f"loudness_{name_suffix}": eval_metric.get_loudness(waveform.numpy()),
        f"panning_{name_suffix}": eval_metric.get_panning(waveform.numpy()),
        f"mid_side_ratio_{name_suffix}": eval_metric.get_mid_side_ratio(waveform.numpy()),
        f"spectral_centroid_{name_suffix}": eval_metric.get_spectral_centroid(waveform.mean(dim=0).numpy()),
        f"band_ratio_{name_suffix}": eval_metric.get_band_ratio(waveform.mean(dim=0).numpy()),
        f"crest_factor_{name_suffix}": eval_metric.get_crest_factor(waveform.mean(dim=0).numpy())
    }
    
    # Add Multi-band Spectral Centroid
    mb_centroids = eval_metric.get_multiband_spectral_centroid(waveform.mean(dim=0).numpy())
    for band, val in mb_centroids.items():
        metrics[f"sc_{band}_{name_suffix}"] = val
        
    return metrics

def load_prior_stats(prior_stats_path):
    with open(prior_stats_path, 'r') as f:
        stats = json.load(f)
    
    # 1. Load Mean (Handle 'mu' key from your json)
    if 'mu' in stats:
        baseline_vec = torch.tensor(stats['mu'], dtype=torch.float32)
    else:
        raise ValueError("Prior stats JSON must contain 'mu' key for mean vector.")
        
    # 2. Load Precision Matrix (Inverse Covariance)
    if 'cov_inv' in stats:
        cov_inv = torch.tensor(stats['cov_inv'], dtype=torch.float32)
        # LogDet(Sigma) = -LogDet(Sigma_Inv)
        cov_logdet = -torch.logdet(cov_inv)
    else:
        raise ValueError("Prior stats JSON must contain 'cov_inv' key for inverse covariance matrix.")
    
    return baseline_vec, cov_inv, cov_logdet

def flatten_params(param_dict):
    """
    Flattens a dictionary of parameter tensors into a single feature vector.
    """
    tensors = []
    # Note: Dictionary iteration order is insertion-ordered in Python 3.7+.
    # This assumes stats were generated with the same order.
    for k, v in param_dict.items():
        # Flatten feature dimensions (anything after Batch and NumTracks)
        if torch.is_tensor(v):
            flat_v = v.view(v.shape[0], v.shape[1], -1)
            tensors.append(flat_v)
            
    if not tensors:
        raise ValueError("Parameter dictionary is empty or contains no tensors.")
    
def logp_x(x, baseline_vec, cov_inv, cov_logdet):
    diff = x - baseline_vec                 
    
    # Use Matrix Multiplication with cov_inv instead of solving linear system
    # norm = diff^T * cov_inv * diff
    # (N, D) @ (D, D) -> (N, D)
    
    # If x is a batch, we need careful dimensions
    if x.ndim == 1:
        norm = diff @ cov_inv @ diff
    else:
        # Batch version: diag(diff @ cov_inv @ diff.T)
        # Efficient way: element-wise multiply and sum
        norm = (diff @ cov_inv * diff).sum(dim=-1)

    return -0.5 * (
        norm + cov_logdet + baseline_vec.shape[0] * math.log(2 * math.pi)
    )

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
    
    # Initialize CLAP Loss for ITO
    try:
        clap_loss_fn = CLAPFeatureLoss(ckpt_path=args.clap_checkpoint)
    except Exception as e:
        print(f"Warning: Could not initialize CLAPFeatureLoss: {e}. ITO might fail.")
        clap_loss_fn = None

    print("Load prior stats for log-prob computation...")
    baseline_vec, cov_inv, cov_logdet = load_prior_stats(args.prior_stats_path)

    all_metrics = []
    all_songs_losses = [] # Store loss history for all songs: list of lists
    
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
        if args.custom_reference is not None:
            mix_filepath = args.custom_reference
            print(f"Using custom reference mix: {mix_filepath}")

        ref_audio, ref_sr = torchaudio.load(mix_filepath, backend="soundfile")
        print(f"Loaded reference mix from {mix_filepath}, SR={ref_sr}, Shape={ref_audio.shape}")
        if ref_sr != 44100:
            ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 44100)
        ref_audio = ref_audio.view(1, 2, -1) # (1, 2, len)
        
        # Determine target track index early to find active slice
        target_idx = args.target_track_idx
        
        # Determine if we need automatic selection (-2)
        auto_select_track = (target_idx == -2)
        
        # Handle Master Bus (-1)
        is_master_control = (target_idx == -1)

        if not is_master_control and not auto_select_track:
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
                if is_master_control or auto_select_track:
                    # Use sum of all tracks (mix proxy) for energy check
                    target_slice = tracks_tensor[0, :, try_idx : try_idx + slice_len].sum(dim=0)
                else:
                    target_slice = tracks_tensor[0, target_idx, try_idx : try_idx + slice_len]
                
                energy = target_slice.pow(2).mean().item()
                
                # if energy > 1e-4: # Threshold for "active"
                #     start_idx = try_idx
                #     found_active = True
                #     break
                
                if energy > best_energy:
                    best_energy = energy
                    best_idx = try_idx
            
            # Use max energy slice
            start_idx = best_idx
            # if not found_active:
            #     print(f"Warning: Could not find active slice for track {target_idx}. Using slice with max energy.")
            #     start_idx = best_idx
        
        # Slice tracks to 10s (same as eval_loop) to avoid OOM
        if start_idx + slice_len > tracks_tensor.shape[-1]:
            start_idx = 0
            if tracks_tensor.shape[-1] < slice_len:
                slice_len = tracks_tensor.shape[-1]
                
        tracks_slice = tracks_tensor[..., start_idx : start_idx + slice_len].clone()
        if args.custom_reference is not None:
            ref_slice = ref_audio[..., 0: slice_len].clone()
        else:
            ref_slice = ref_audio[..., start_idx : start_idx + slice_len].clone()

        # --- Select top 8 active tracks ---
        # Calculate energy of each track in the slice
        track_energies = tracks_slice.squeeze(0).pow(2).mean(dim=-1) # (num_tracks,)
        
        if auto_select_track:
             target_idx = torch.argmax(track_energies).item()
             print(f"Auto-selected highest energy track index: {target_idx}")
        
        # We must include target_idx if it's a specific track
        selected_indices = []
        if not is_master_control:
            selected_indices.append(target_idx)
        
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
        if not is_master_control:
            target_idx = selected_indices.index(target_idx)
        else:
            target_idx = -1 # Keep as -1 for master control
            
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
        print("Baseline inference done.")
            
        # --- Step 2: Text Prompt on Target Track (ITO) ---
        # Target track index determined earlier

        # track_idx, text_alpha, style_alpha, text_prompt, is_panning = text
        
        # NOTE: Text Interpolation method commented out in favor of ITO
        # if args.num_iterations > 0 and False:
        #     text_input = (target_idx, args.text_alpha, args.style_alpha, args.text_prompt, args.is_panning)
        #     ... (original logic) ...
        
        if args.ito_num_step > 0 and clap_loss_fn is not None:
            print(f"Running ITO for {args.ito_num_step} steps on track {target_idx} with prompt '{args.text_prompt[0]}'")
            
            # Prepare for ITO
            prompt_str = args.text_prompt[0] 
            neg_str = args.neg_prompt[0] if args.neg_prompt is not None else None
            bs, num_tracks, seq_len = pred_tracks_base.shape[0], pred_tracks_base.shape[2], pred_tracks_base.shape[3]
            
            # --- WHOLE MIX ITO ADJUSTMENT ---
            # If target_idx == -1 (Master), we want to optimize the MIX embedding, not track embeddings.
            # But full_base_embedding above is computed from SEPARATE tracks. 
            # If we want to optimize the mix, we should encode the MIX.
            
            if is_master_control:

                print(f"[INFO] pred_mix_base shape: {pred_mix_base.shape}")
                
                ito_embedding = model.mix_encoder(pred_mix_base)
                print(f"[INFO] ito_embedding shape: {ito_embedding.shape}")
                fit_embedding = torch.nn.Parameter(ito_embedding, requires_grad=True)
                optimizer = torch.optim.RAdam([fit_embedding], lr=args.ito_lr)
                
            else:
                # Encode Separate Tracks
                full_input = pred_tracks_base.clone().view(bs, num_tracks * 2, -1)
                full_base_embedding = model.mix_encoder(full_input)
                # full_base_embedding: (bs, 2*num_tracks, embed_dim)
                full_base_embedding = full_base_embedding.view(bs, num_tracks * 2, -1)
                
                num_tracks_mix = num_tracks
                effective_target_idx = target_idx

                full_base_embedding = full_base_embedding.detach()
                
                target_L = full_base_embedding[0:1, effective_target_idx : effective_target_idx + 1, :]
                target_R = full_base_embedding[0:1, effective_target_idx + num_tracks_mix : effective_target_idx + num_tracks_mix + 1, :]
                
                print(f"[INFO] Target L shape: {target_L.shape}, Target R shape: {target_R.shape}")
                
                initial_reference_feature = torch.cat([target_L, target_R], dim=1)
                print(f"[INFO] Initial reference feature shape: {initial_reference_feature.shape}")
                
                fit_embedding = torch.nn.Parameter(initial_reference_feature, requires_grad=True)
                optimizer = torch.optim.RAdam([fit_embedding], lr=args.ito_lr) 
                print(f"[INFO] Fitting embedding shape: {fit_embedding.shape}")
                
                # [Corrected] Construct ito_embedding using Masking
                base = full_base_embedding.clone().detach() 
                print(f"[INFO] Base embedding shape: {base.shape}")
                
                fit_expanded = torch.zeros_like(base)
                fit_expanded[0, effective_target_idx, :] = fit_embedding[0, 0, :]
                fit_expanded[0, effective_target_idx + num_tracks_mix, :] = fit_embedding[0, 1, :]
                
                mask = torch.zeros_like(base)
                mask[0, effective_target_idx, :] = 1.0
                mask[0, effective_target_idx + num_tracks_mix, :] = 1.0
                
                ito_embedding = (fit_expanded * mask) + (base * (1 - mask))
            
            # Initialize reference for mixing 
            curr_ref_mix = pred_mix_base.detach()
            curr_ref_tracks = pred_tracks_base.detach()
            
            min_loss = float('inf')
            min_loss_step = 0
            best_results = {
                "mix": pred_mix_base,
                "tracks": pred_tracks_base,
                "track_params": pred_track_params,
                "fx_params": pred_fx_params,
                "master_params": pred_master_params
            }
            
            song_losses = []
            
            for ito_step in range(args.ito_num_step):
                optimizer.zero_grad()
                
                # [ITO Logic] Construct ito_embedding from the learnable parameter
                if is_master_control:
                    # For Full Mix, we optimize the whole embedding directly
                    ito_embedding = fit_embedding
                else:
                    # For Track Control, we mask the specific track into the static base
                    # Use masking to preserve gradients from fit_embedding
                    base_static = full_base_embedding.detach()
                    
                    fit_expanded = torch.zeros_like(base_static)
                    fit_expanded[0, effective_target_idx, :] = fit_embedding[0, 0, :]
                    fit_expanded[0, effective_target_idx + num_tracks_mix, :] = fit_embedding[0, 1, :]
                    
                    mask = torch.zeros_like(base_static)
                    mask[0, effective_target_idx, :] = 1.0
                    mask[0, effective_target_idx + num_tracks_mix, :] = 1.0
                    
                    ito_embedding = (fit_expanded * mask) + (base_static * (1 - mask))

                # Prepare Reference Audio
                if is_master_control:
                    ref_audio_input = batch_stereo_peak_normalize(curr_ref_mix)
                else:
                    norm_tracks = batch_stereo_tracks_peak_normalize(curr_ref_tracks)
                    bs_ref, chs_ref, num_tracks_ref, len_ref = norm_tracks.shape
                    # Flatten for run_diffmst
                    ref_audio_input = norm_tracks.view(bs_ref, chs_ref*num_tracks_ref, -1)
                
                # Run Model
                if ito_step == 0:
                    prev_t, prev_f, prev_m = pred_track_params, pred_fx_params, pred_master_params
                else:
                    prev_t, prev_f, prev_m = best_results["track_params"], best_results["fx_params"], best_results["master_params"]
                
                result = run_diffmst(
                    tracks_slice,
                    ref_audio_input, # Detached previous output
                    model,
                    mix_console,
                    text=None, # No text interpolation
                    interpolation=args.interpolation,
                    track_start_idx=0,
                    ref_start_idx=0,
                    ito_embedding=ito_embedding, # Pass the optimized embedding
                    # prev_track_param_dict=prev_t,
                    # prev_fx_bus_param_dict=prev_f,
                    # prev_master_bus_param_dict=prev_m,
                    use_master_bus=True
                )
                
                (pred_mix_ito, pred_tracks_ito, p_track, p_fx, p_master) = result
                
                # Compute Loss
                # Loss on target track (mono)
                if is_master_control:
                    target_audio = pred_mix_ito
                else:
                    target_audio = pred_tracks_ito[:, :, target_idx, :]
                
                # Mix to mono for CLAP
                target_mono = target_audio.mean(dim=1, keepdim=True)
                
                print(p_track)
                loss = clap_loss_fn(target_mono, target=prompt_str, neg_target=neg_str, sample_rate=44100, distance_fn="cosine") - logp_x(flatten_params(p_track), baseline_vec, cov_inv, cov_logdet).mean()*args.prior_loss_weight
                
                if not loss.requires_grad:
                    print("!! CRITICAL ERROR: Loss does not require grad. computational graph is broken anywhere.")
                
                loss.backward()
                
                if fit_embedding.grad is None:
                    print("!! CRITICAL ERROR: fit_embedding.grad is None. Backprop didn't reach the parameter.")
                
                prev_embedding = fit_embedding.clone().detach()
                optimizer.step()
                
                # param_change = (fit_embedding - prev_embedding).abs().sum().item()
                # grad_norm = fit_embedding.grad.norm().item() if fit_embedding.grad is not None else 0.0
                # if param_change == 0 and grad_norm > 0:
                #     print("!! WARNING: Parameters did not change despite having gradients. Check learning rate.")
                    
                loss_val = loss.item()
                song_losses.append(loss_val)
                
                if loss_val < min_loss:
                    min_loss = loss_val
                    min_loss_step = ito_step
                    best_results = {
                        "mix": pred_mix_ito.detach(),
                        "tracks": pred_tracks_ito.detach(),
                        "track_params": p_track,
                        "fx_params": p_fx,
                        "master_params": p_master
                    }

                if ito_step % 5 == 0 or ito_step == args.ito_num_step - 1:
                    print(f"ITO Step {ito_step+1}/{args.ito_num_step}, CLAP Loss: {loss_val:.4f}, Min Loss: {min_loss:.4f} at step {min_loss_step}")
        
            all_songs_losses.append(song_losses)
            
            # Use best results
            pred_mix_text = best_results["mix"]
            pred_tracks_text = best_results["tracks"]
            pred_track_params = best_results["track_params"]
            pred_fx_params = best_results["fx_params"]
            pred_master_params = best_results["master_params"]
            
            print(f"Best ITO loss: {min_loss:.4f} at step {min_loss_step}")
            
            # Save Loss Curve for this song
            song_out_dir = output_dir / song_name
            song_out_dir.mkdir(exist_ok=True)
            
            plt.figure(figsize=(10, 6))
            plt.plot(song_losses, label=f'{song_name}')
            plt.title(f'ITO Loss Curve - {song_name}')
            plt.xlabel('Step')
            plt.ylabel('CLAP Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(song_out_dir / "loss_curve.png")
            plt.close()
            
        else:
            if args.ito_num_step > 0:
                 print("Skipping ITO because CLAP loss not initialized.")
            pred_mix_text = None
            pred_tracks_text = None

        # --- Sum Mix ---
        sum_mix = tracks_slice.sum(dim=1, keepdim=True).repeat(1, 2, 1) # (1, 2, len)

        # Baseline
        if is_master_control:
            target_stem_base = pred_mix_base[0] # (2, len)
            sum_others_base = torch.zeros_like(target_stem_base)
        else:
            target_stem_base = pred_tracks_base[0, :, target_idx, :] # (2, len)
            other_stems_base = pred_tracks_base[0].clone()
            other_stems_base[:, target_idx, :] = 0
            sum_others_base = other_stems_base.sum(dim=1) # (2, len)
        
        if args.ito_num_step > 0 and pred_mix_text is not None:
            # Text
            if is_master_control:
                target_stem_text = pred_mix_text[0]
                sum_others_text = torch.zeros_like(target_stem_text)
            else:
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

        if args.ito_num_step > 0 and pred_mix_text is not None:
            metrics["ITO_best_loss"] = min_loss
            metrics.update(compute_audio_metrics(target_stem_text, "target_text_modified"))
            metrics.update(compute_audio_metrics(sum_others_text, "others_text_modified"))

        # Compute AF Loss
        # pred_mix_base: (1, 2, len)
        # ref_slice: (1, 2, len)
        af_losses_base = af_loss_fn(pred_mix_base, ref_slice)
        if args.ito_num_step > 0 and pred_mix_text is not None:
            af_losses_text = af_loss_fn(pred_mix_text, ref_slice)
        af_losses_sum = af_loss_fn(sum_mix, ref_slice)

        # Calculate and log Total AF Loss (sum of all components)
        metrics["AF_base_total"] = sum(af_losses_base.values()).item()
        for k, v in af_losses_base.items():
            metrics[f"AF_base_{k}"] = v.item()

        if args.ito_num_step > 0 and pred_mix_text is not None:
            metrics["AF_text_total"] = sum(af_losses_text.values()).item()
            for k, v in af_losses_text.items():
                metrics[f"AF_text_{k}"] = v.item()

            # CLAP Similarity
            # Try to access CLAP model from the loaded model's text_encoder
            if hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'model'):
                clap_model = model.text_encoder.model
                
                if is_master_control:
                    # For master control, compare text with the full mix (sum of tracks)
                    origin_target = tracks_slice[0].sum(dim=0) # (2, len)
                else:
                    origin_target = tracks_slice[0, target_idx, :]
                
                pred_target = target_stem_text
                
                metrics["CLAP_text_target_origin"] = compute_clap_similarity(clap_model, origin_target, args.text_prompt[0])
                metrics["CLAP_text_target_pred"] = compute_clap_similarity(clap_model, pred_target, args.text_prompt[0])
            else:
                print("Warning: Could not find CLAP model in text_encoder. Skipping CLAP metrics.")

        metrics["AF_sum_total"] = sum(af_losses_sum.values()).item()
        for k, v in af_losses_sum.items():
            metrics[f"AF_sum_{k}"] = v.item()

        # --- Loudness Normalization ---
        # Normalize mixes to target LUFS
        pred_mix_base, pred_tracks_base = normalize_audio(pred_mix_base, pred_tracks_base, args.target_lufs, meter, "baseline")
        if args.ito_num_step > 0 and pred_mix_text is not None:
            pred_mix_text, pred_tracks_text = normalize_audio(pred_mix_text, pred_tracks_text, args.target_lufs, meter, "text")
        
        dummy_stems_sum = tracks_slice.unsqueeze(1).repeat(1, 2, 1, 1)
        sum_mix, _ = normalize_audio(sum_mix, dummy_stems_sum, args.target_lufs, meter, "sum")

        # --- Save Audio ---
        song_out_dir = output_dir / song_name
        song_out_dir.mkdir(exist_ok=True)
        
        # Save Mixes
        torchaudio.save(song_out_dir / "mix_baseline.wav", pred_mix_base.squeeze(0), 44100)
        if args.ito_num_step > 0 and pred_mix_text is not None:
            torchaudio.save(song_out_dir / "mix_ito_text.wav", pred_mix_text.squeeze(0), 44100)
        torchaudio.save(song_out_dir / "mix_sum.wav", sum_mix.squeeze(0), 44100)
        
        # Normalize stems for audibility (Note: this changes relative mix balance in the saved file)
        target_stem_base = normalize_stem(target_stem_base, args.target_lufs, meter, "target_base")
        sum_others_base = normalize_stem(sum_others_base, args.target_lufs, meter, "others_base")

        torchaudio.save(song_out_dir / "target_baseline.wav", target_stem_base, 44100)
        torchaudio.save(song_out_dir / "others_baseline.wav", sum_others_base, 44100)
        
        if args.ito_num_step > 0 and pred_mix_text is not None:
            target_stem_text = normalize_stem(target_stem_text, args.target_lufs, meter, "target_text")
            sum_others_text = normalize_stem(sum_others_text, args.target_lufs, meter, "others_text")

            torchaudio.save(song_out_dir / "target_ito_text.wav", target_stem_text, 44100)
            torchaudio.save(song_out_dir / "others_ito_text.wav", sum_others_text, 44100)
            
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

    # --- Plot Average Loss Curve ---
    if args.ito_num_step > 0 and all_songs_losses:
        try:
            # list of lists -> (num_songs, num_iterations)
            losses_arr = np.array(all_songs_losses) 
            # Check dimensions
            if losses_arr.ndim == 2:
                avg_losses = np.mean(losses_arr, axis=0)
                
                plt.figure(figsize=(10, 6))
                
                # Plot individual songs faintly
                for s_idx, s_losses in enumerate(all_songs_losses):
                    plt.plot(range(len(s_losses)), s_losses, alpha=0.15, color='gray')
                    
                plt.plot(range(len(avg_losses)), avg_losses, label='Average Loss', color='blue', linewidth=2)
                
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Average CLAP Loss Optimization Curve')
                plt.legend()
                plt.grid(True)
                plt.savefig(output_dir / "avg_loss_curve.png")
                plt.close()
                print(f"Saved average loss curve to {output_dir / 'avg_loss_curve.png'}")
            else:
                 print(f"Skipping average loss plot: Inconsistent loss array shape {losses_arr.shape}")
            
        except Exception as e:
            print(f"Could not plot average loss curve: {e}")

if __name__ == "__main__":
    main()
