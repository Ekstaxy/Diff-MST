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
from tqdm import tqdm
import torch.nn.functional as F

# Add parent directory to path to import mst
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mst.utils import load_diffmst, batch_stereo_peak_normalize, batch_stereo_tracks_peak_normalize

def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (torch.Tensor, np.ndarray)):
        if hasattr(obj, 'numel') and obj.numel() == 1:
            return obj.item()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
    return obj

def parse_args():
    parser = argparse.ArgumentParser(description='Run generic track prior extraction on a dataset')
    
    # Model configs
    parser.add_argument("--config", type=str, required=True, help='Path to model config (e.g. configs/models/naive.yaml)')
    parser.add_argument("--checkpoint", type=str, required=True, help='Path to model checkpoint')
    
    # Dataset configs
    parser.add_argument("--dataset_root", type=str, required=True, help='Root directory of the dataset')
    parser.add_argument("--num_songs", type=int, default=50, help='Number of songs to process for statistics')
    parser.add_argument("--seed", type=int, default=42, help='Random seed')
    parser.add_argument("--output_file", type=str, default="track_prior_stats.json", help='Path to save the JSON output')
    
    return parser.parse_args()

def collect_track_params(args, model):
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Find all songs
    print(f"Scanning dataset at {args.dataset_root}...")
    song_paths = []
    
    # Heuristic to find song folders: Look for folders containing .wav files
    # A robust way is to walk the directory
    for root, dirs, files in os.walk(args.dataset_root):
        # specific to current dataset structure: <song>/RAW/<tracks>
        # Check if this folder has .wav files
        wav_files = [f for f in files if f.endswith(".wav")]
        if len(wav_files) > 1:
            # Check if it looks like a song folder (e.g. ends with RAW)
            if os.path.basename(root) == "RAW":
                # Check if MIX exists in parent
                mix_path = os.path.join(os.path.dirname(root), os.path.basename(os.path.dirname(root)) + "_MIX.wav")
                # Or maybe inconsistent naming.
                # Let's just store the relative path
                rel_path = os.path.relpath(root, args.dataset_root)
                song_paths.append(rel_path)
    
    # Sort and shuffle to random selection
    song_paths.sort()
    random.shuffle(song_paths)
    
    selected_songs = song_paths[:args.num_songs]
    print(f"Selected {len(selected_songs)} songs for statistics calculation.")
    
    model.eval()
    
    all_track_params_list = []
    
    for song_idx, song_rel_path in enumerate(tqdm(selected_songs)):
        song_dir = os.path.join(args.dataset_root, song_rel_path)
        
        # 1. Load Data (Tracks and Mix)
        # ----------------------------------------
        track_filepaths = glob.glob(os.path.join(song_dir, "*.wav"))
        
        # Try to find the mix file
        # Parent of RAW usually contains the mix
        parent_dir = os.path.dirname(song_dir)
        mix_files = glob.glob(os.path.join(parent_dir, "*_MIX.wav"))
        
        if not mix_files:
             # Fallback: look inside the current dir if maybe it's mixed
             # But usually structure is Song/RAW and Song/Song_MIX.wav
             continue
             
        mix_filepath = mix_files[0]
        
        # Load Reference Mix
        try:
            ref_audio, ref_sr = torchaudio.load(mix_filepath, backend="soundfile")
        except:
            continue
            
        if ref_sr != 44100:
            ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 44100)
            
        # Ensure stereo
        if ref_audio.shape[0] == 1:
            ref_audio = ref_audio.repeat(2, 1)
        ref_audio = ref_audio.view(1, 2, -1) # (1, 2, Len)
        
        # Load Tracks
        tracks = []
        lengths = []
        valid_tracks = True
        
        for track_filepath in track_filepaths:
            try:
                audio, sr = torchaudio.load(track_filepath, backend="soundfile")
            except:
                valid_tracks = False
                break
                
            if sr != 44100:
                audio = torchaudio.functional.resample(audio, sr, 44100)
            if audio.shape[0] == 2:
                audio = audio.mean(dim=0, keepdim=True) # Mix to mono for track input
            
            tracks.append(audio)
            lengths.append(audio.shape[-1])
            
        if not valid_tracks or len(tracks) == 0:
            continue
            
        max_length = max(lengths)
        
        # Pad tracks
        for idx in range(len(tracks)):
            tracks[idx] = F.pad(tracks[idx], (0, max_length - lengths[idx]))
            
        tracks_tensor = torch.cat(tracks, dim=0) # (NumTracks, Len)
        tracks_tensor = tracks_tensor.unsqueeze(0) # (1, NumTracks, Len)
        
        # Align lengths
        if ref_audio.shape[-1] != max_length:
            # Crop to min
            min_len = min(ref_audio.shape[-1], max_length)
            ref_audio = ref_audio[..., :min_len]
            tracks_tensor = tracks_tensor[..., :min_len]
        
        # Slice to a manageable length (e.g. 10s) to run model
        # Just pick the middle 10s or beginning
        slice_len = 44100 * 10
        if tracks_tensor.shape[-1] > slice_len:
            start = (tracks_tensor.shape[-1] - slice_len) // 2
            tracks_analyze = tracks_tensor[..., start : start + slice_len]
            ref_analyze = ref_audio[..., start : start + slice_len]
        else:
            tracks_analyze = tracks_tensor
            ref_analyze = ref_audio

        # Batch Normalize (Important for model)
        # Using the same normalization functions as training/eval
        try:
            norm_tracks = batch_stereo_tracks_peak_normalize(tracks_analyze.unsqueeze(1).repeat(1, 2, 1, 1).view(*tracks_analyze.shape[:2], 2, -1))
            # norm_tracks comes out as (BS, Ch, Num, Len) -> We need (BS, Num, Len) for encoder if mono?
            # Actually Diff-MST takes (BS, 2, Num, Len) or (BS, Num, Len)?
            # mst/utils.py line 125/126 implies norm_analysis_tracks is cleaned before passing.
            # But wait, model.track_encoder usually expects (BS, NumTracks, Len).
            # Let's check model.track_encoder in modules.py?
            # Actually, let's just use what run_diffmst does logic-wise but manually.
            
            # Re-read utils.py:
            # norm_tracks = [] ... for track_idx ...
            # it constructs norm_analysis_tracks.
            
            # Let's perform simple peak norm
            tracks_analyze = tracks_analyze / (tracks_analyze.abs().max(dim=-1, keepdim=True)[0] + 1e-8) * 0.5 
            ref_analyze = ref_analyze / (ref_analyze.abs().max(dim=-1, keepdim=True)[0] + 1e-8) * 0.5
            
        except Exception as e:
            print(f"Error normalizing: {e}")
            continue

        # 2. Run Model Inference
        # ----------------------
        with torch.no_grad():
            # (A) Encode Tracks
            # track_encoder expects (bs, num_tracks, seq_len)
            bs, num_tracks, seq_len = tracks_analyze.shape
            
            # Flatten track batch for encoder if necessary, or pass directly depending on implementation
            # Examining modules.py usually:
            # track_enc = self.track_encoder(tracks)
            # if tracks is (BS, Num, Len), it might expect reshaping. 
            # Usually Diff-MST track_encoder flattens internally or expects (BS*Num, 1, Len).
            
            # Let's assume standard behavior:
            # We treat each track as an independent sample for the encoder
            flat_tracks = tracks_analyze.view(bs * num_tracks, 1, seq_len)
            track_embeds = model.track_encoder(flat_tracks) # (BS*Num, EmbedDim)
            
            # Reshape back: (BS, Num, EmbedDim)
            track_embeds = track_embeds.view(bs, num_tracks, -1)
            
            # (B) Encode Reference Mix
            # mix_encoder expects (BS, 2, Len)
            mix_embeds = model.mix_encoder(ref_analyze) # (BS, 2, EmbedDim) - or something similar
            
            # (C) Controller -> Predict Parameters
            # controller(track_embeds, mix_embeds)
            
            # We obtain "pred_track_params"
            # Return shape: (BS, NumTracks, ParamDim)
            pred_track_params, _, _ = model.controller(track_embeds, mix_embeds)
            
            # 3. Collect Data
            # Flatten to (NumTracks, ParamDim)
            flat_params = pred_track_params.view(-1, pred_track_params.shape[-1])
            all_track_params_list.append(flat_params.cpu())

    if len(all_track_params_list) == 0:
        print("No data collected.")
        return

    # 4. Compute Statistics
    # ---------------------
    all_data = torch.cat(all_track_params_list, dim=0) # (TotalSamples, ParamDim)
    print(f"Collected total {all_data.shape[0]} parameter samples.")
    
    # Mean
    mu = torch.mean(all_data, dim=0)
    
    # Covariance
    # cov(m) expects (Variables, Observations), so we transpose
    cov = torch.cov(all_data.T)
    
    # Inverse Covariance (Precision Matrix)
    # Add epsilon for numerical stability
    epsilon = 1e-6 * torch.eye(cov.shape[0]).to(cov.device)
    cov_inv = torch.inverse(cov + epsilon)
    
    print("Computed Mu and Cov Inverse.")
    
    # 5. Save to JSON
    # ---------------
    output_data = {
        "mu": mu.tolist(),
        "cov_inv": cov_inv.tolist()
    }
    
    # Serializing
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Saved statistics to {args.output_file}")


def main():
    args = parse_args()
    
    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = load_diffmst(args.config, args.checkpoint)
    model.eval()
    
    collect_track_params(args, model)

if __name__ == "__main__":
    main()
