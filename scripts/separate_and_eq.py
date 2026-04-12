import os
import torch
import torchaudio
import torchaudio.functional as F
from pathlib import Path
from tqdm import tqdm

# 引入您的分離模組 (請確認路徑與您的專案相符)
import sys
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

def generate_separated_stems(
    mix_path, 
    output_dir, 
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
    讀取 Mix 音檔 -> 裁切 -> RoFormer 分離 -> (可選) 套用 EQ -> 轉單聲道存檔
    """
    mix_path = Path(mix_path)
    if not mix_path.exists():
        print(f"[錯誤] 找不到檔案: {mix_path}")
        return

    # 1. 載入並重採樣
    mix_audio, sr = torchaudio.load(mix_path)
    if sr != target_sr:
        mix_audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(mix_audio)
    
    # 2. 裁切 (Crop)
    start_idx = int(start_sec * target_sr)
    length_idx = int(length_sec * target_sr)
    
    if start_idx + length_idx > mix_audio.shape[-1]:
        print(f"[警告] {mix_path.name} 長度不足，將盡可能擷取。")
        mix_audio = mix_audio[..., start_idx:]
    else:
        mix_audio = mix_audio[..., start_idx : start_idx + length_idx]

    # [防呆機制] 確保餵給分離模型的 Mix 是 Stereo (2 Channels)
    if mix_audio.shape[0] == 1:
        mix_audio_stereo = mix_audio.repeat(2, 1) # 複製成雙聲道
    else:
        mix_audio_stereo = mix_audio

    # 3. 執行 Source Separation
    # 擴充 batch 維度: (1, 2, seq_len)
    batch_mix = mix_audio_stereo.unsqueeze(0).to(device) 
    
    with torch.no_grad():
        # 假設輸出形狀為 (Batch, Sources, Channels, Time)
        # 依據您原本的 Code: separated_sources[0, 1] 是 Vocals, [0, 0] 是 Inst
        separated_sources = separator(batch_mix)
        vocal_est = separated_sources[0, 1].cpu()
        inst_est = separated_sources[0, 0].cpu()

    # 原本輸入的裁切版 Mix 也轉回 CPU 準備存檔
    mix_audio = mix_audio.cpu()

    # 4. 套用 EQ (若開啟，則對分離出來的 Stems 處理)
    if apply_eq:
        vocal_est = apply_eq_filter(vocal_est, target_sr, filter_type, cutoff_freq)
        inst_est = apply_eq_filter(inst_est, target_sr, filter_type, cutoff_freq)
        eq_suffix = f"_EQ_{filter_type}{int(cutoff_freq)}"
    else:
        eq_suffix = "_NoEQ"

    # 5. 強制轉為單聲道 (Mono) 以符合推論架構
    if vocal_est.shape[0] > 1: vocal_est = vocal_est.mean(dim=0, keepdim=True)
    if inst_est.shape[0] > 1: inst_est = inst_est.mean(dim=0, keepdim=True)
    if mix_audio.shape[0] > 1: mix_audio = mix_audio.mean(dim=0, keepdim=True)

    # 6. 建立輸出資料夾與存檔
    song_name = mix_path.stem # 取得檔名 (不含副檔名)
    save_dir = Path(output_dir) / f"{song_name}{eq_suffix}"
    save_dir.mkdir(parents=True, exist_ok=True)

    torchaudio.save(save_dir / f"{song_name}_mix_input.wav", mix_audio, target_sr)
    torchaudio.save(save_dir / f"{song_name}_vocal_sep.wav", vocal_est, target_sr)
    torchaudio.save(save_dir / f"{song_name}_inst_sep.wav", inst_est, target_sr)
    
    print(f"[+] 處理完成: {save_dir.name}")

# =====================================================================
# 本地端執行區塊 (讓您可以自由放入 List 測試)
# =====================================================================
if __name__ == "__main__":
    
    # --- 1. 基本設定 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ROFORMER_CKPT = "model_bs_roformer_ep_317_sdr_12.9755.ckpt" # 替換成您的路徑
    OUTPUT_ROOT = "./test_separated_stems"
    
    # --- 2. 載入分離模型 ---
    print("[*] 正在載入 RoFormer 分離模型...")
    separator = RoFormerRemixer(sample_rate=44100, model_name=ROFORMER_CKPT).to(DEVICE)
    separator.eval()
    
    # --- 3. 定義您要測試的 Mix 音檔清單 ---
    my_mix_files = [
        "/work/your_path/song_1_mix.wav",
        "/work/your_path/song_2_mix.wav"
        # 隨時可以在這加檔案
    ]
    
    # --- 4. 執行大迴圈！ ---
    for mix_file in tqdm(my_mix_files, desc="Separating files"):
        
        # 測試 A: 正常分離 (不加 EQ)，擷取 0~6 秒
        generate_separated_stems(
            mix_path=mix_file,
            output_dir=OUTPUT_ROOT,
            separator=separator,
            device=DEVICE,
            start_sec=0.0,
            length_sec=6.0,
            apply_eq=False
        )
        
        # 測試 B: 分離後套用 100Hz 低通濾波 (Low-pass)，擷取 0~6 秒
        generate_separated_stems(
            mix_path=mix_file,
            output_dir=OUTPUT_ROOT,
            separator=separator,
            device=DEVICE,
            start_sec=0.0,
            length_sec=6.0,
            apply_eq=True,
            filter_type="lowpass",
            cutoff_freq=100.0
        )
        
        # 測試 C: 分離後套用 8000Hz 高通濾波 (High-pass)，擷取 0~6 秒
        generate_separated_stems(
            mix_path=mix_file,
            output_dir=OUTPUT_ROOT,
            separator=separator,
            device=DEVICE,
            start_sec=0.0,
            length_sec=6.0,
            apply_eq=True,
            filter_type="highpass",
            cutoff_freq=8000.0
        )