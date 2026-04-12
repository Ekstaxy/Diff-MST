import sys
# sys.path = [p for p in sys.path if not p.startswith('/usr/local/lib')]
sys.modules['flash_attn'] = None
import os
import torch
import torchaudio
import torchaudio.functional as F
from pathlib import Path
from tqdm import tqdm

# 引入您的分離模組 (請確認路徑)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mst.modules import RoFormerRemixer

def apply_eq_filter(audio, sr, filter_type, cutoff_freq):
    """套用高通或低通濾波器"""
    if filter_type == 'lowpass':
        return F.lowpass_biquad(audio, sample_rate=sr, cutoff_freq=cutoff_freq)
    elif filter_type == 'highpass':
        return F.highpass_biquad(audio, sample_rate=sr, cutoff_freq=cutoff_freq)
    else:
        raise ValueError(f"不支援的濾波器類型: {filter_type}")

def load_crop_resample(file_path, target_sr, start_idx, length_idx):
    """輔助函數：負責讀取、重採樣與精準裁切音檔"""
    if not file_path.exists():
        return None
        
    audio, sr = torchaudio.load(file_path)
    
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(audio)
        
    # 裁切 (防呆：如果長度不夠就擷取到底)
    if start_idx + length_idx > audio.shape[-1]:
        audio = audio[..., start_idx:]
    else:
        audio = audio[..., start_idx : start_idx + length_idx]
        
    return audio

def process_single_song_folder(
    song_dir, 
    output_root, 
    separator, 
    device="cuda",
    start_sec=0.0, 
    length_sec=6.0, 
    apply_eq=False, 
    filter_type="lowpass", 
    cutoff_freq=100.0,
    target_sr=44100
):
    """
    讀取 MUSDB 資料夾 -> 分離 Mixture -> 合併 Bass+Drums+Other -> 套用 EQ -> 轉單聲道存檔
    """
    song_dir = Path(song_dir)
    if not song_dir.exists() or not song_dir.is_dir():
        print(f"[錯誤] 找不到資料夾: {song_dir}")
        return

    # 預期擁有的檔案清單
    mix_path = song_dir / "mixture.wav"
    voc_path = song_dir / "vocals.wav"
    bass_path = song_dir / "bass.wav"
    drums_path = song_dir / "drums.wav"
    other_path = song_dir / "other.wav"

    if not mix_path.exists():
        print(f"[警告] {song_dir.name} 缺少 mixture.wav，跳過此資料夾。")
        return

    start_idx = int(start_sec * target_sr)
    length_idx = int(length_sec * target_sr)

    # ==========================================
    # 1. 處理 Ground Truth (真實軌道)
    # ==========================================
    gt_vocal = load_crop_resample(voc_path, target_sr, start_idx, length_idx)
    gt_bass = load_crop_resample(bass_path, target_sr, start_idx, length_idx)
    gt_drums = load_crop_resample(drums_path, target_sr, start_idx, length_idx)
    gt_other = load_crop_resample(other_path, target_sr, start_idx, length_idx)

    # 確保 stems 都有讀到，合成真實的 Instrumental
    if gt_bass is not None and gt_drums is not None and gt_other is not None:
        gt_inst = gt_bass + gt_drums + gt_other
    else:
        print(f"[警告] {song_dir.name} 缺少部分分軌，無法建立 Ground Truth，跳過。")
        return

    # ==========================================
    # 2. 處理 Mixture 與 Source Separation
    # ==========================================
    mix_audio = load_crop_resample(mix_path, target_sr, start_idx, length_idx)

    # [防呆機制] 分離模型 (RoFormer) 必須吃雙聲道 (Stereo)
    if mix_audio.shape[0] == 1:
        mix_audio_stereo = mix_audio.repeat(2, 1)
    else:
        mix_audio_stereo = mix_audio

    batch_mix = mix_audio_stereo.unsqueeze(0).to(device) 
    
    with torch.no_grad():
        separated_sources = separator(batch_mix)
        # 依據 RoFormer 輸出：[0, 1] 是 Vocals, [0, 0] 是 Instrumental
        sep_vocal = separated_sources[0, 1].cpu()
        sep_inst = separated_sources[0, 0].cpu()

    # ==========================================
    # 3. 套用 EQ (僅對分離出來的 Stems 套用)
    # ==========================================
    if apply_eq:
        sep_vocal = apply_eq_filter(sep_vocal, target_sr, filter_type, cutoff_freq)
        sep_inst = apply_eq_filter(sep_inst, target_sr, filter_type, cutoff_freq)
        eq_suffix = f"_EQ_{filter_type}{int(cutoff_freq)}"
    else:
        eq_suffix = "_Normal"

    # ==========================================
    # 4. 強制轉換為單聲道 (Mono) 供後續推論使用
    # ==========================================
    def to_mono(tensor):
        return tensor.mean(dim=0, keepdim=True) if tensor.shape[0] > 1 else tensor

    mix_audio = to_mono(mix_audio)
    sep_vocal = to_mono(sep_vocal)
    sep_inst = to_mono(sep_inst)
    gt_vocal = to_mono(gt_vocal)
    gt_inst = to_mono(gt_inst)

    # ==========================================
    # 5. 建立資料夾並儲存所有檔案
    # ==========================================
    song_name = song_dir.name
    save_dir = Path(output_root) / f"{song_name}{eq_suffix}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 存檔：Mixture 輸入
    torchaudio.save(save_dir / f"{song_name}_mix_input.wav", mix_audio, target_sr)
    
    # 存檔：分離出來的軌道 (可能帶有 EQ)
    torchaudio.save(save_dir / f"{song_name}_vocal_sep.wav", sep_vocal, target_sr)
    torchaudio.save(save_dir / f"{song_name}_inst_sep.wav", sep_inst, target_sr)
    
    # 存檔：真實軌道 (完美音質，用作上限對照組)
    torchaudio.save(save_dir / f"{song_name}_vocal_gt.wav", gt_vocal, target_sr)
    torchaudio.save(save_dir / f"{song_name}_inst_gt.wav", gt_inst, target_sr)
    
    print(f"[+] 處理完成: {save_dir.name}")

# =====================================================================
# 本地端執行區塊 (自由定義 List 跑迴圈)
# =====================================================================
if __name__ == "__main__":
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ROFORMER_CKPT = "model_bs_roformer_ep_317_sdr_12.9755.ckpt" # 請確認您的 ckpt 路徑
    OUTPUT_ROOT = "/work/ajchen2005/processed_eval_dataset" # 所有處理完的資料夾都會放在這
    
    print("[*] 正在載入 RoFormer 分離模型...")
    separator = RoFormerRemixer(sample_rate=44100, model_name=ROFORMER_CKPT).to(DEVICE)
    separator.eval()
    
    # --- 在這裡放入您的 MUSDB 資料夾清單 ---
    my_song_folders = [
        "/work/ajchen2005/musdb18hq/test/Al James - Schoolboy Facination",
        "/work/ajchen2005/musdb18hq/test/Angels In Amplifiers - I'm Alright",
        "/work/ajchen2005/musdb18hq/test/Cristina Vane - So Easy"
        # 自由新增...
    ]
    
    for folder_path in tqdm(my_song_folders, desc="Processing Song Folders"):
        
        # 測試情境 1：正常分離 (無 EQ)，從第 30 秒開始擷取 6 秒
        process_single_song_folder(
            song_dir=folder_path,
            output_root=OUTPUT_ROOT,
            separator=separator,
            device=DEVICE,
            start_sec=60.0,     # [您要求的參數] 開始秒數   
            length_sec=16.0,     # [您要求的參數] 長度秒數
            apply_eq=False      # [您要求的 flag] 是否開啟 EQ
        )
        
        # 測試情境 2：分離 + 100Hz 低通濾波 (Low-pass)
        process_single_song_folder(
            song_dir=folder_path,
            output_root=OUTPUT_ROOT,
            separator=separator,
            device=DEVICE,
            start_sec=60.0,
            length_sec=16.0,
            apply_eq=True,
            filter_type="lowpass", # [您要求的參數] 高通或低通
            cutoff_freq=500.0      # [您要求的參數] 濾波頻率
        )
        
        # 測試情境 3：分離 + 8000Hz 高通濾波 (High-pass)
        process_single_song_folder(
            song_dir=folder_path,
            output_root=OUTPUT_ROOT,
            separator=separator,
            device=DEVICE,
            start_sec=60.0,
            length_sec=16.0,
            apply_eq=True,
            filter_type="highpass",
            cutoff_freq=2000.0
        )


# inference_folder = "/work/ajchen2005/test_songs"
# all_song_dirs = [d for d in Path(inference_folder).iterdir() if d.is_dir()]
# # make all .wav ends with track_inst.wav rename to track_inst.wav
# # make all .wav ends with track_vocal.wav rename to track_vocal.wav
# # make all .wav ends with ref_inst.wav rename to ref_inst.wav
# # make all .wav ends with ref_vocal.wav rename to ref_vocal.wav
# for song_dir in all_song_dirs:
#     for wav_file in song_dir.glob("*.wav"):
#         print(f"Processing {wav_file.name} in {song_dir.name}")
#         if wav_file.name.endswith("track_inst.wav"):
#             new_name = "track_inst.wav"
#             wav_file.rename(song_dir / new_name)
#         elif wav_file.name.endswith("track_vocal.wav"):
#             new_name = "track_vocal.wav"
#             wav_file.rename(song_dir / new_name)
#         elif wav_file.name.endswith("ref_inst.wav"):
#             new_name = "ref_inst.wav"
#             wav_file.rename(song_dir / new_name)
#         elif wav_file.name.endswith("ref_vocal.wav"):
#             new_name = "ref_vocal.wav"
#             wav_file.rename(song_dir / new_name)
#         elif wav_file.name.endswith("mix_input.wav"):
#             new_name = "mix.wav"
#             wav_file.rename(song_dir / new_name)

#         # print(f"Renamed {wav_file.name} to {new_name} in {song_dir.name}")
