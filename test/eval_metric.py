import librosa
import numpy as np
import argparse
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Do evaluation of mixing metrics')

    # Audio parameters
    parser.add_argument("--tracks_path", type=str, default="inference_result/", help='Path to folder with processed audio tracks')
    
    # # Other parameters
    # parser.add_argument("--output_dir", type=str, default="./eval_outputs",
    #                     help='Directory to save evaluation outputs')
    # parser.add_argument("--exp_name", type=str, default="test", 
    #                     help='Experiment name for output folder')

    return parser.parse_args()

def get_spectral_centroid(waveform, sr=44100): 
    """
    Compute the average spectral centroid of the given waveform.
    Args:
        waveform (np.ndarray): Audio waveform of dimension (Channels, time)
    Returns:
        np.ndarray: Spectral centroid of dimension (Channels,)
    """
    # Ensure 2D
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
        
    centroids = []
    for y in waveform:
        cent = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            window="hann"
        )
        centroids.append(np.mean(cent))
    
    return np.array(centroids)

def get_band_ratio(waveform, sr=44100, split_freq=1000):
    """
    Compute the ratio of energy in high frequency bands to low frequency bands.
    Args:
        waveform (np.ndarray): Audio waveform of dimension (Channels, time)
    Returns:
        np.ndarray: Band ratio of dimension (Channels,)
    """
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]

    ratios = []
    for y in waveform:
        # Power spectrogram
        S = np.abs(librosa.stft(
            y,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            window="hann"
        ))**2
        
        freq_per_bin = sr / 2048
        split_bin = int(split_freq / freq_per_bin)
        
        low_band_energy = np.sum(S[:split_bin, :])
        high_band_energy = np.sum(S[split_bin:, :])
        
        ratio = high_band_energy / (low_band_energy + 1e-8)
        ratios.append(ratio)
        
    return np.array(ratios)

def get_crest_factor(waveform):
    """
    Compute the Crest Factor of the given waveform.
    Args:
        waveform (np.ndarray): Audio waveform of dimension (Channels, time)
    Returns:
        np.ndarray: Crest Factor of dimension (Channels,)
    """
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
        
    # Peak: max absolute value along time axis
    peak = np.max(np.abs(waveform), axis=-1)
    
    # RMS: root mean square along time axis
    rms = np.sqrt(np.mean(waveform**2, axis=-1))
    
    # Crest Factor (dB)
    cf_linear = peak / (rms + 1e-8)
    cf_db = 20 * np.log10(cf_linear)
    
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
    
    file_list = []
    if os.path.isdir(args.tracks_path):
        file_list = glob.glob(os.path.join(args.tracks_path, "*.wav"))
    else:
        file_list = [args.tracks_path]
        
    all_results = []
    
    for file_path in file_list:
        print(f"Processing: {file_path}")
        try:
            waveform, sr = librosa.load(file_path, sr=44100, mono=False)
            metrics = foward(waveform)
            
            # Add filename to metrics for identification
            metrics['filename'] = os.path.basename(file_path)
            all_results.append(metrics)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    print("\nAll Results:")
    for res in all_results:
        print(res['filename'])
        print(f"  Spectral Centroid: {res['spectral_centroid']}")
        print(f"  Band Ratio: {res['band_ratio']}")
        print(f"  Crest Factor: {res['crest_factor']}")