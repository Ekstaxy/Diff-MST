import os
import glob
import yaml
import torch
import torchaudio
import argparse
import random
import numpy as np
import soundfile as sf
from tqdm import tqdm
from pathlib import Path

# Update importing path to include root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mst.modules import AdvancedMixConsole, RoFormerRemixer
from mst.mixing import naive_random_mix

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-process dataset: Mix -> Separate -> Save")
    parser.add_argument("--config", type=str, default="configs/data/musdb18-2.yaml", help="Path to data config")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the processed dataset")
    parser.add_argument("--augmentations", type=int, default=10, help="How many random mixes per song?")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--duration", type=float, default=20.0, help="Duration in seconds per clip")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--roformer_model", type=str, default="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    return parser.parse_args()

def load_metadata(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract root dirs and metadata files from the config structure
    # Assuming structure: data -> init_args -> track_root_dirs / metadata_files
    try:
        init_args = config['data']['init_args']
        track_root_dirs = init_args['track_root_dirs']
        metadata_files = init_args['metadata_files']
    except KeyError:
        # try flat structure if simplified
        track_root_dirs = config.get('track_root_dirs', [])
        metadata_files = config.get('metadata_files', [])
        
    return track_root_dirs, metadata_files

def get_song_dirs(track_root_dirs, metadata_files):
    # This logic mimics MultitrackDataModule
    song_dirs = {} # keys: dir path, values: list of valid track filenames
    
    # Load metadata allowed files
    allowed_files = set()
    for meta_file in metadata_files:
        # meta_file path might be relative to project root
        if not os.path.exists(meta_file):
             # Try relative to config location or project root logic
             # Assuming running from project root
             pass
        
        with open(meta_file, 'r') as f:
            meta = yaml.safe_load(f)
            # Flatten structure to get list of files
            # The structure in musdb18.yaml is typically song_name: [file1, file2...] or flat list?
            # Based on checking dataloader, it seems it maps dir to files.
            # Let's assume metadata is dict of {song_name: {tracks: [...]}} or similar
            # Or simpler: just walk directories and filter.
            pass
            # For now, let's rely on directory walking as fallback if metadata is complex
            
    # Simple walk strategy (robust fallback)
    all_song_paths = []
    for root_dir in track_root_dirs:
        # e.g. /content/musdb18hq/train/Song Name/
        # or /content/musdb18hq/Song Name/
        # Recursive search for folders containing wav files
        for root, dirs, files in os.walk(root_dir):
            wavs = [f for f in files if f.endswith('.wav') and not f.startswith('._')]
            if len(wavs) >= 2: # At least 2 stems to mix
                all_song_paths.append(root)
                
    return sorted(list(set(all_song_paths)))

def process_song(song_dir, args, mixer, separator, output_root):
    song_name = os.path.basename(song_dir)
    # Create output dir for this song
    save_dir = os.path.join(output_root, song_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load Tracks
    wav_files = glob.glob(os.path.join(song_dir, "*.wav"))
    tracks = []
    
    # Identify track names (metadata)
    # Simple logic: load all valid wavs except 'mixture.wav'
    loaded_audio = []
    
    length_samples = int(args.duration * args.sample_rate)
    
    # Determine common length of full song
    max_len = 0
    valid_files = []
    for f in wav_files:
        if "mixture.wav" in f: continue
        info = torchaudio.info(f)
        if info.num_frames > max_len:
            max_len = info.num_frames
        valid_files.append(f)
        
    if max_len < length_samples:
        print(f"Skipping {song_name}: too short ({max_len} < {length_samples})")
        return

    # Load all full tracks into RAM (they are usually manageable)
    # We will slice them randomly later
    full_tracks = []
    for f in valid_files:
        audio, sr = torchaudio.load(f)
        # Resample if needed
        if sr != args.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, args.sample_rate)
            audio = resampler(audio)
        
        # Mono/Stereo check - force Mono for mixer input? 
        # The system usually expects (bs, num_tracks, seq_len) where tracks are mono
        # If input is stereo, we might need to mixdown or keep stereo?
        # Looking at system.py: "ref_mix_mid = ref_mix.sum(dim=1)" -> ref mix is stereo
        # Input tracks to mixer: usually mono stems.
        if audio.shape[0] == 2:
            audio = audio.mean(dim=0, keepdim=True) # Convert stem to mono for simple mixing
        
        full_tracks.append(audio)
    
    if not full_tracks:
        return

    full_tracks_tensor = torch.stack(full_tracks).to(args.device) # (num_tracks, 1, full_len)
    full_tracks_tensor = full_tracks_tensor.squeeze(1) # (num_tracks, full_len)
    
    # 2. Augmentation Loop
    for i in range(args.augmentations):
        # Random Crop
        if full_tracks_tensor.shape[-1] > length_samples:
            start = random.randint(0, full_tracks_tensor.shape[-1] - length_samples)
            current_slice = full_tracks_tensor[:, start:start+length_samples]
        else:
            # Pad if exactly equal or slight mismatch
            current_slice = full_tracks_tensor[:, :length_samples] # handle later
            if current_slice.shape[-1] < length_samples:
                # Pad
                pad_amt = length_samples - current_slice.shape[-1]
                current_slice = torch.nn.functional.pad(current_slice, (0, pad_amt))
        
        # Add Batch Dim for Mixer: (1, num_tracks, seq_len)
        batch_slice = current_slice.unsqueeze(0)
        
        # Random Mix
        # This calls naive_random_mix which generates random gains/EQ/Pan
        # Returns: mixed_tracks, mix, param_dicts...
        # mix shape: (bs, 2, seq_len) -> Stereo Mix
        (
            mixed_tracks, 
            ref_mix, 
            track_param_dict, 
            fx_bus_param_dict, 
            master_bus_param_dict, 
            mix_params, 
            fx_bus_params, 
            master_bus_params
        ) = naive_random_mix(
            batch_slice, 
            mixer,
            use_track_input_fader=True,
            use_track_panner=True,
            use_track_eq=True, # Random EQ
            use_track_compressor=True,
            use_fx_bus=True,
            use_master_bus=True
        )
        
        # Normalize Mix
        ref_mix_max = ref_mix.abs().max()
        if ref_mix_max > 0:
            ref_mix = ref_mix / (ref_mix_max + 1e-8) * 0.9 # Peak normalize to -1dB roughly
        
        # 3. Source Separation (The slow part)
        # RoFormerRemixer expects (bs, 2, seq_len)
        try:
            # separated_sources shape: (bs, 2, 2, seq_len) -> (Batch, Stems[Inst, Voc], Stereo, Time)
            separated_sources = separator(ref_mix) 
        except Exception as e:
            print(f"Separation failed for {song_name} aug {i}: {e}")
            continue
            
        # 4. Save to Disk
        # Filename pattern: {song_name}_aug{i}
        base_name = f"aug_{i}"
        
        # Save Mix
        mix_path = os.path.join(save_dir, f"{base_name}_mix.wav")
        save_audio(ref_mix[0], mix_path, args.sample_rate)
        
        # Save Separated Vocals
        # separated_sources[0, 1] is Vocals (based on RoFormerRemixer logic order [Inst, Voc])
        vocab_est = separated_sources[0, 1]
        voc_path = os.path.join(save_dir, f"{base_name}_vocals_est.wav")
        save_audio(vocab_est, voc_path, args.sample_rate)
        
        # Save Separated Instrumental
        # separated_sources[0, 0] is Instrumental
        inst_est = separated_sources[0, 0]
        inst_path = os.path.join(save_dir, f"{base_name}_other_est.wav")
        save_audio(inst_est, inst_path, args.sample_rate)
        
        # Save Metadata / Parameters (Optional, for training inputs)
        # We might need the original 'tracks' (mono stems) as input to the model?
        # System.py: "tracks = separated_tracks" -> The model input IS the separated stems.
        # But we also need 'ref_mix' (done).
        # We also need 'target parameters' (mix_params) if we are training to predict them?
        # System.py: "ref_track_param_dict... = ref_params" (Ground Truth)
        # Yes, you need to save the Ground Truth parameters so the model can learn to predict them!
        
        params_path = os.path.join(save_dir, f"{base_name}_params.pt")
        torch.save({
            'track_params': mix_params[0].cpu(),
            'fx_bus_params': fx_bus_params[0].cpu(),
            'master_bus_params': master_bus_params[0].cpu(),
            'mixed_tracks': mixed_tracks[0].cpu() # Original mono stems processed
        }, params_path)

def save_audio(tensor, path, sr):
    # tensor: (channels, time)
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    data = tensor.numpy().T # (time, channels)
    sf.write(path, data, sr)

def main():
    args = parse_args()
    
    print(f"Setting up dataset generation...")
    print(f"Output Dir: {args.output_dir}")
    print(f"Augmentations per song: {args.augmentations}")
    
    # 1. Setup Modules
    mixer = AdvancedMixConsole(sample_rate=args.sample_rate).to(args.device)
    separator = RoFormerRemixer(sample_rate=args.sample_rate, model_name=args.roformer_model).to(args.device)
    
    # 2. Get Data List
    track_root_dirs, metadata_files = load_metadata(args.config)
    song_dirs = get_song_dirs(track_root_dirs, metadata_files)
    
    print(f"Found {len(song_dirs)} songs in {track_root_dirs}")
    
    # 3. Process
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
