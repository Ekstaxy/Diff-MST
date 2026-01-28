import numpy as np

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

def linear_interpolation(emb1: np.ndarray, emb2: np.ndarray, alpha: float) -> np.ndarray:
    """ Linear interpolation between two embeddings. """
    return (1 - alpha) * emb1 + alpha * emb2

def spherical_linear_interpolation(emb1: np.ndarray, emb2: np.ndarray, alpha: float) -> np.ndarray:
    """ Spherical linear interpolation between two embeddings. """
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)

    dot_product = np.dot(emb1_norm, emb2_norm)
    omega = np.arccos(np.clip(dot_product, -1.0, 1.0))
    sin_omega = np.sin(omega)

    if sin_omega < 1e-6:  # Fall back to linear interpolation
        return linear_interpolation(emb1, emb2, alpha)

    factor1 = np.sin((1 - alpha) * omega) / sin_omega
    factor2 = np.sin(alpha * omega) / sin_omega

    new_emb = factor1 * emb1 + factor2 * emb2

    return new_emb