import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import torch
import pathlib
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Do evaluation of mixing metrics')

    # Audio parameters
    parser.add_argument("--tracks_path", type=str, 
                        default="/mnt/gestalt/home/rakec/data/diff-mst/MedleyDB_V2/V2/TleilaxEnsemble_Late/TleilaxEnsemble_Late_RAW",
                        help='Path to folder with processed audio tracks')
    
    # Other parameters
    parser.add_argument("--output_dir", type=str, default="./eval_outputs",
                        help='Directory to save evaluation outputs')
    parser.add_argument("--exp_name", type=str, default="test", 
                        help='Experiment name for output folder')

    return parser.parse_args()

def get_spectral_centroid(waveform): 
    """
    Compute the average spectral centroid of the given waveform.
    Args:
        waveform (Tensor): Audio waveform of dimension (..., time)
    Returns:
        Tensor: Spectral centroid of dimension (...,)
    """
    centroid = F.spectral_centroid(
        waveform = waveform,
        sample_rate = 44100,
        pad = 0,
        window = "hann",
        n_fft = 2048,
        win_length = 2048,
        hop_length = 512
    )
    return centroid.mean(dim=-1)

def get_band_ratio(waveform, split_freq=1000):
    """
    Compute the ratio of energy in high frequency bands to low frequency bands.
    Args:
        waveform (Tensor): Audio waveform of dimension (..., time)
    Returns:
        Tensor: Band ratio of dimension (...,)
    """

    spec_transform = T.Spectrogram(
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        power=2.0
    )

    spec = spec_transform(waveform) # Shape: (Batch, Freq_bins, Time)
        
    freq_per_bin = 44100 / 2048
    split_bin = int(split_freq / freq_per_bin)

    low_band_energy = spec[..., :split_bin, :].sum(dim=(-2, -1))
    high_band_energy = spec[..., split_bin:, :].sum(dim=(-2, -1))

    ratio = high_band_energy / (low_band_energy + 1e-8)
    return ratio

def get_crest_factor(waveform):
        """
        Compute the Crest Factor of the given waveform.
        Args:
            waveform (Tensor): Audio waveform of dimension (..., time)
        Returns:
            Tensor: Crest Factor of dimension (...,)
        """
        # 1. Peak: 找出絕對值的最大值
        # dim=-1 代表沿著時間軸找
        peak = waveform.abs().max(dim=-1)[0]
        
        # 2. RMS: 均方根
        rms = torch.sqrt(waveform.pow(2).mean(dim=-1))
        
        # 3. 計算 Crest Factor (dB)
        cf_linear = peak / (rms + 1e-8)
        cf_db = 20 * torch.log10(cf_linear)
        
        return cf_db

def foward(waveform):
    # Example function to demonstrate usage of the above metrics
    spectral_centroid = get_spectral_centroid(waveform)
    band_ratio = get_band_ratio(waveform)
    crest_factor = get_crest_factor(waveform)

    return {
        "spectral_centroid": spectral_centroid, 
        "band_ratio": band_ratio, 
        "crest_factor": crest_factor
    }

if __name__ == "__main__":
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    #