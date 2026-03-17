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
    # 將 augmentations 數量訂在參數中，你可以透過指令或修改這裡的預設值來當作 config
    parser.add_argument("--augmentations", type=int, default=10, help="How many random mixes per song?")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--duration", type=float, default=20.0, help="Duration in seconds per clip")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--roformer_model", type=str, default="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    return parser.parse_args()

def load_metadata(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    try:
        init_args = config['data']['init_args']
        track_root_dirs = init_args.get('track_root_dirs', [])
        metadata_files = init_args.get('metadata_files', [])
    except KeyError:
        track_root_dirs = config.get('track_root_dirs', [])
        metadata_files = config.get('metadata_files', [])
        
    return track_root_dirs, metadata_files

def get_song_dirs(track_root_dirs, metadata_files):
    all_song_paths = []
    # 嚴格定義 MUSDB18 必須擁有的檔案
    expected_stems = {'vocals.wav', 'bass.wav', 'drums.wav', 'other.wav'}
    
    for root_dir in track_root_dirs:
        for root, dirs, files in os.walk(root_dir):
            # 只有當這 4 個檔案都存在於該資料夾時，才加進去
            if expected_stems.issubset(set(files)):
                all_song_paths.append(root)
                
    return sorted(list(set(all_song_paths)))

def process_song(song_dir, args, mixer, separator, output_root):
    song_name = os.path.basename(song_dir)
    save_dir = os.path.join(output_root, song_name)
    os.makedirs(save_dir, exist_ok=True)
    
    existing_mixes = glob.glob(os.path.join(save_dir, "*_mix.wav"))
    if len(existing_mixes) >= args.augmentations:
        return
    
    start_aug_idx = 0
    if len(existing_mixes) > 0:
        indices = [int(os.path.basename(f).split('_')[1]) for f in existing_mixes if '_' in os.path.basename(f)]
        if indices:
            start_aug_idx = max(indices) + 1
            
    # 1. 讀取並辨識單軌 (MUSDB18 格式: vocals, bass, drums, other)
    # 我們強制定義順序，確保 Model 輸入一致
    expected_stems = ['vocals.wav', 'bass.wav', 'drums.wav', 'other.wav']
    loaded_stems = {}
    
    length_samples = int(args.duration * args.sample_rate)
    max_len = 0
    
    for stem_name in expected_stems:
        file_path = os.path.join(song_dir, stem_name)
        if os.path.exists(file_path):
            audio, sr = torchaudio.load(file_path)
            if sr != args.sample_rate:
                audio = torchaudio.transforms.Resample(sr, args.sample_rate)(audio)
            
            # 強制轉為單聲道 (Mono) 以符合系統輸入
            if audio.shape[0] == 2:
                audio = audio.mean(dim=0, keepdim=True)
                
            loaded_stems[stem_name] = audio
            max_len = max(max_len, audio.shape[-1])
            
    if 'vocals.wav' not in loaded_stems:
        print(f"Skipping {song_name}: No vocals.wav found.")
        return
        
    if max_len < length_samples:
        print(f"Skipping {song_name}: too short.")
        return

    # 2. Augmentation Loop
    target_total = args.augmentations
    
    for i in range(start_aug_idx, target_total):
        
        # --- [重點 1] 人聲能量偵測 (VAD) 與安全裁切 ---
        valid_crop = False
        start_idx = 0
        vocal_track = loaded_stems['vocals.wav']
        
        # 嘗試 50 次找到有聲音的 20 秒
        for _ in range(50):
            start = random.randint(0, max_len - length_samples)
            voc_slice = vocal_track[:, start:start+length_samples]
            
            # 計算這 20 秒人聲的 RMS 能量
            rms = torch.sqrt(torch.mean(voc_slice ** 2))
            dbfs = 20 * torch.log10(rms + 1e-8)
            
            if dbfs > -35.0:  # 設定 -35 dB 為靜音閥值 (可依需求調整)
                valid_crop = True
                start_idx = start
                break
                
        if not valid_crop:
            print(f"Warning: Could not find valid vocal segment for {song_name} aug {i}. Skipping this aug.")
            continue
        
        # --- [重點 2] 準備未處理的 Track 與合成 Dry Instrument ---
        current_slices = []
        for stem_name in expected_stems:
            if stem_name in loaded_stems:
                current_slices.append(loaded_stems[stem_name][:, start_idx:start_idx+length_samples])
            else:
                current_slices.append(torch.zeros((1, length_samples))) # 缺少的軌道補 0
                
        # 分離出你想存的 Dry Tracks (保持在 CPU 以利儲存)
        dry_vocal = current_slices[0] # Vocals
        dry_instrumental = current_slices[1] + current_slices[2] + current_slices[3] # Bass + Drums + Other
        
        # [修改處] 將 Vocal 與合併後的 Instrumental 轉到 GPU，並組合給 Mixer
        # 這會讓輸入 Mixer 的軌道數從 4 變成 2
        vocal_tensor = dry_vocal.to(args.device)
        instrumental_tensor = dry_instrumental.to(args.device)
        
        # 組合 Tensor: (2, seq_len)
        two_tracks_tensor = torch.cat([vocal_tensor, instrumental_tensor], dim=0)
        batch_slice = two_tracks_tensor.unsqueeze(0) # 形狀變成 (1, 2, seq_len)
        
        # --- Random Mix (產生 random 參數與立體聲混音) ---
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
            use_track_eq=True,
            use_track_compressor=True,
            use_fx_bus=True,
            use_master_bus=True
        )
        
        # Normalize Mix
        ref_mix_max = ref_mix.abs().max()
        if ref_mix_max > 0:
            ref_mix = ref_mix / (ref_mix_max + 1e-8) * 0.9 
            
        # --- [重點 3] Source Separation ---
        try:
            # 加入 no_grad 防止 OOM
            with torch.no_grad():
                separated_sources = separator(ref_mix) 
        except Exception as e:
            print(f"Separation failed for {song_name} aug {i}: {e}")
            torch.cuda.empty_cache()
            continue
            
        # 4. Save to Disk
        base_name = f"aug_{i}"
        
        # 4.1 儲存你的需求：Dry Tracks
        save_audio(dry_vocal, os.path.join(save_dir, f"{base_name}_dry_vocal.wav"), args.sample_rate)
        save_audio(dry_instrumental, os.path.join(save_dir, f"{base_name}_dry_instrumental.wav"), args.sample_rate)
        
        # 4.2 儲存 Reference Mix
        mix_path = os.path.join(save_dir, f"{base_name}_mix.wav")
        save_audio(ref_mix[0], mix_path, args.sample_rate)
        
        # 4.3 儲存分離出來的軌道
        vocab_est = separated_sources[0, 1]
        voc_path = os.path.join(save_dir, f"{base_name}_vocals_est.wav")
        save_audio(vocab_est, voc_path, args.sample_rate)
        
        inst_est = separated_sources[0, 0]
        inst_path = os.path.join(save_dir, f"{base_name}_other_est.wav")
        save_audio(inst_est, inst_path, args.sample_rate)
        
        # 4.4 儲存 Parameters 與裁切紀錄
        params_path = os.path.join(save_dir, f"{base_name}_params.pt")
        torch.save({
            'crop_start': start_idx,               # 紀錄切在原始音檔的哪個 index
            'track_params': mix_params[0].cpu(),
            'fx_bus_params': fx_bus_params[0].cpu(),
            'master_bus_params': master_bus_params[0].cpu(),
            'dry_vocal': dry_vocal.cpu(),          # 也可以選擇把波形包在 pt 裡，讀取更快
            'dry_instrumental': dry_instrumental.cpu()
        }, params_path)

        # 釋放 GPU 記憶體
        del separated_sources
        torch.cuda.empty_cache()

def save_audio(tensor, path, sr):
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    data = tensor.numpy().T 
    sf.write(path, data, sr)

def main():
    args = parse_args()
    
    print(f"Setting up dataset generation...")
    print(f"Output Dir: {args.output_dir}")
    print(f"Augmentations per song: {args.augmentations}")
    
    mixer = AdvancedMixConsole(sample_rate=args.sample_rate).to(args.device)
    separator = RoFormerRemixer(sample_rate=args.sample_rate, model_name=args.roformer_model).to(args.device)
    separator.eval() # 確保分離模型在 eval 模式
    
    track_root_dirs, metadata_files = "/work/ajchen2005/musdb18hq", "/home/ajchen2005/Diff-MST/configs/data/musdb18-2.yaml"
    song_dirs = get_song_dirs(track_root_dirs, metadata_files)
    
    print(f"Found {len(song_dirs)} songs in {track_root_dirs}")
    
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