# run pretrained models over evaluation set to generate audio examples for the listening test
import os
import torch
import torchaudio
import pyloudnorm as pyln
import argparse
from mst.utils import load_diffmst, run_diffmst
from mst.loss import compute_barkspectrum, compute_rms, compute_crest_factor, compute_stereo_width, compute_stereo_imbalance, AudioFeatureLoss
import json
import numpy as np
import csv


def equal_loudness_mix(tracks: torch.Tensor, *args, **kwargs):

    meter = pyln.Meter(44100)
    target_lufs_db = -48.0

    norm_tracks = []
    for track_idx in range(tracks.shape[1]):
        track = tracks[:, track_idx : track_idx + 1, :]
        lufs_db = meter.integrated_loudness(track.squeeze(0).permute(1, 0).numpy())

        if lufs_db < -80.0:
            print(f"Skipping track {track_idx} with {lufs_db:.2f} LUFS.")
            continue

        lufs_delta_db = target_lufs_db - lufs_db
        track *= 10 ** (lufs_delta_db / 20)
        norm_tracks.append(track)

    norm_tracks = torch.cat(norm_tracks, dim=1)
    # create a sum mix with equal loudness
    sum_mix = torch.sum(norm_tracks, dim=1, keepdim=True).repeat(1, 2, 1)
    sum_mix /= sum_mix.abs().max()

    return sum_mix, None, None, None

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Diff-MST model')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Computation device')
    
    # Model paths
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Audio paths
    parser.add_argument('--tracks', type=str, required=True,
                        help='Path to tracks folder')
    parser.add_argument('--reference', type=str, required=True,
                        help='Path to reference mix WAV file')
    
    # Optional settings
    parser.add_argument('--output', type=str, default='outputs/listen',
                        help='Output directory')
    parser.add_argument('--name', type=str, default='experiment',
                        help='Experiment name')
    parser.add_argument('--target-lufs', type=float, default=-22.0,
                        help='Target output LUFS')
    parser.add_argument('--ref-lufs', type=float, default=-16.0,
                        help='Reference LUFS')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    meter = pyln.Meter(44100)
    target_lufs_db = args.target_lufs
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        "diffmst-16": {
            "model": load_diffmst(
                args.config,
                args.checkpoint,
                map_location=args.device,
            ),
            "func": run_diffmst,
        },
        "sum": {
            "model": (None, None),
            "func": equal_loudness_mix,
        },
    }

    # get the validation examples
    examples = {
        # "ecstasy": {
        #     "tracks": "/Users/svanka/Downloads//diffmst-examples/song1/BenFlowers_Ecstasy_Full/",
        #     "ref": "/Users/svanka/Downloads//diffmst-examples/song1/ref/_Feel it all Around_ by Washed Out (Portlandia Theme)_01.wav",
        # },
        # "by-my-side": {
        #     "tracks": "/Users/svanka/Downloads//diffmst-examples/song2/Kat Wright_By My Side/",
        #     "ref": "/Users/svanka/Downloads//diffmst-examples/song2/ref/The Dip - Paddle To The Stars (Lyric Video)_01.wav",
        # },
        "haunted-aged": {
            "tracks": args.tracks,
            "ref": args.reference,
        },
    }
    loss = AudioFeatureLoss([0.1,0.001,1.0,1.0,0.1], 44100)
    AF = {}
    #initialise to negative infinity
   
    for example_name, example in examples.items():

        AF[example_name] = {}
        print(example_name)
        example_dir = os.path.join(output_dir, example_name)
        os.makedirs(example_dir, exist_ok=True)
        json_dir = os.path.join(output_dir, "AF")
        if not os.path.exists(json_dir):
            os.makedirs(json_dir, exist_ok=True)
        csv_path = os.path.join(json_dir,f"{example_name}.csv")
        # if not os.path.exists(csv_path):
        #     os.makedirs(csv_path)
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["method", "audio_section","track_start_idx", "track_stop_idx", "ref_start_idx", "ref_stop_idx", "rms", "crest_factor", "stereo_width", "stereo_imbalance", "barkspectrum", "net_AF_loss"])
            f.close()

        # ----------load reference mix---------------
        ref_audio, ref_sr = torchaudio.load(example["ref"], backend="soundfile")
        if ref_sr != 44100:
            ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 44100)
        print(ref_audio.shape, ref_sr)
        ref_length = ref_audio.shape[-1]
        # --------------first find all the tracks----------------
        track_filepaths = []
        for root, dirs, files in os.walk(example["tracks"]):
            for filepath in files:
                if filepath.endswith(".wav"):
                    track_filepaths.append(os.path.join(root, filepath))

        print(f"Found {len(track_filepaths)} tracks.")

        # ----------------load the tracks----------------------------
        tracks = []
        lengths = []
        for track_idx, track_filepath in enumerate(track_filepaths):
            audio, sr = torchaudio.load(track_filepath, backend="soundfile")

            if sr != 44100:
                audio = torchaudio.functional.resample(audio, sr, 44100)

            # loudness normalize the tracks to -48 LUFS
            lufs_db = meter.integrated_loudness(audio.permute(1, 0).numpy())
            # lufs_delta_db = -48 - lufs_db
            # audio = audio * 10 ** (lufs_delta_db / 20)

            print(track_idx, os.path.basename(track_filepath), audio.shape, sr, lufs_db)

            if audio.shape[0] == 2:
                audio = audio.mean(dim=0, keepdim=True)

            chs, seq_len = audio.shape

            for ch_idx in range(chs):
                tracks.append(audio[ch_idx : ch_idx + 1, :])
                lengths.append(audio.shape[-1])

        # find max length and pad if shorter
        max_length = max(lengths)
        min_length = min(lengths)   
        for track_idx in range(len(tracks)):
            tracks[track_idx] = torch.nn.functional.pad(
                tracks[track_idx], (0, max_length - lengths[track_idx])
            )

        # stack into a tensor
        tracks = torch.cat(tracks, dim=0)
        tracks = tracks.view(1, -1, max_length)
        ref_audio = ref_audio.view(1, 2, -1)

        # crop tracks to max of 60 seconds or so
        # tracks = tracks[..., :4194304]
        tracks_length = max_length
        
        #print(tracks.shape)
        track_start_idx = int(tracks_length / 4)
        ref_start_idx = int(ref_length / 4)
        track_stop_idx = int(3*tracks_length / 4)
        ref_stop_idx = int(3*ref_length / 4)
        #find the number of sets of track samples of 10 sec duration each
        track_num_sets = int((track_stop_idx - track_start_idx) / 441000)
        ref_num_sets = int((ref_stop_idx - ref_start_idx) / 441000)
        print("track_num_sets", track_num_sets)
        print("ref_num_sets", ref_num_sets)
        min_AF_loss = float('inf')
        min_AF_loss_example = None
        for i in range(track_num_sets):
            for j in range(ref_num_sets):
                print(f"track-{i}-ref-{j}")
                #run inference for every combination of track and ref samples and calculate audio features. 
                # We will save the audio features to a csv and audio files in the output directory
                mix_tracks = tracks[..., track_start_idx + i*441000 : track_start_idx + (i+1)*441000]
                ref_analysis = ref_audio[..., ref_start_idx + j*441000 : ref_start_idx + (j+1)*441000]

                # create mixes varying the loudness of the reference
                for ref_loudness_target in [args.ref_lufs]:  # -16, -20, -24, -28
                    print("Ref loudness", ref_loudness_target)
                    ref_filepath = os.path.join(
                        example_dir,
                        f"ref-analysis-track-{i}-ref-{j}-lufs-{ref_loudness_target:0.0f}.wav",
                    )

                    # loudness normalize the reference mix section to -14 LUFS
                    ref_lufs_db = meter.integrated_loudness(
                        ref_analysis.squeeze().permute(1, 0).numpy()
                    )
                    print("ref_lufs_db", ref_lufs_db)
                    lufs_delta_db = ref_loudness_target - ref_lufs_db
                    ref_analysis = ref_analysis * 10 ** (lufs_delta_db / 20)

                    torchaudio.save(ref_filepath, ref_analysis.squeeze(), 44100)
                    
                    AF_loss = 0
                    for method_name, method in methods.items():
                        AF[example_name][method_name] = {}
                        print(method_name)
                        # tracks (torch.Tensor): Set of input tracks with shape (bs, num_tracks, seq_len)
                        # ref_audio (torch.Tensor): Reference mix with shape (bs, 2, seq_len)

                        if method_name == "sum":
                            if ref_loudness_target != -16:
                                continue


                        model, mix_console = method["model"]
                        func = method["func"]

                        #print(tracks.shape, ref_audio.shape)
                        audio_section = f"track-{i}-ref-{j}-lufs-{ref_loudness_target:0.0f}"
                        AF[example_name][method_name][audio_section] = {}
                        AF[example_name][method_name][audio_section]["track_start_idx"] = track_start_idx + i*441000
                        AF[example_name][method_name][audio_section]["track_stop_idx"] = track_start_idx + (i+1)*441000
                        AF[example_name][method_name][audio_section]["ref_start_idx"] = ref_start_idx + j*441000
                        AF[example_name][method_name][audio_section]["ref_stop_idx"] = ref_start_idx + (j+1)*441000

                        with torch.no_grad():
                            result = func(
                                mix_tracks.clone(),
                                ref_analysis.clone(),
                                model.to(args.device),
                                mix_console.to(args.device),
                                track_start_idx=0,
                                ref_start_idx=0,
                            )

                            (
                                pred_mix,
                                pred_track_param_dict,
                                pred_fx_bus_param_dict,
                                pred_master_bus_param_dict,
                            ) = result

                        bs, chs, seq_len = pred_mix.shape
                        print("pred_mix shape", pred_mix.shape)
                        # loudness normalize the output mix
                        mix_lufs_db = meter.integrated_loudness(
                            pred_mix.squeeze(0).permute(1, 0).numpy()
                        )
                        print("pred_mix_lufs_db", mix_lufs_db)
                        #print(mix_lufs_db)
                        lufs_delta_db = target_lufs_db - mix_lufs_db
                        pred_mix = pred_mix * 10 ** (lufs_delta_db / 20)
                        mix_filepath = os.path.join(
                            example_dir,
                            f"{example_name}-{method_name}-tracks-{i}-ref={j}-lufs-{ref_loudness_target:0.0f}.wav",
                        )
                        torchaudio.save(mix_filepath, pred_mix.view(chs, -1), 44100)
                        
                        # compute audio features
                        AF_loss = loss(pred_mix, ref_analysis)
                       
                        for key, value in AF_loss.items():
                            AF[example_name][method_name][audio_section][key] = value.detach().cpu().numpy()
                        AF[example_name][method_name][audio_section]["net_AF_loss"]  = sum(AF_loss.values()).detach().cpu().numpy()
                        print(AF[example_name][method_name][audio_section])

                        if AF[example_name][method_name][audio_section]["net_AF_loss"]  < min_AF_loss:
                            min_AF_loss = AF[example_name][method_name][audio_section]["net_AF_loss"]
                            min_AF_loss_example = f"{example_name}-{method_name}-{audio_section}"
                        print("min_AF_loss", min_AF_loss)
                        print("min_AF_loss_example", min_AF_loss_example)
                        # save resulting audio and parameters
                        #append to csv the method name, audio section, audio features values and net loss on different columns
                       
                        with open(csv_path, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([method_name, audio_section, AF[example_name][method_name][audio_section]["track_start_idx"], AF[example_name][method_name][audio_section]["track_stop_idx"], AF[example_name][method_name][audio_section]["ref_start_idx"], AF[example_name][method_name][audio_section]["ref_stop_idx"], AF[example_name][method_name][audio_section]["mix-rms"], AF[example_name][method_name][audio_section]["mix-crest_factor"], AF[example_name][method_name][audio_section]["mix-stereo_width"], AF[example_name][method_name][audio_section]["mix-stereo_imbalance"], AF[example_name][method_name][audio_section]["mix-barkspectrum"], AF[example_name][method_name][audio_section]["net_AF_loss"]])
                            f.close()
                      
        
        print(f"for {example_name} min loss is {min_AF_loss} corresponding to {min_AF_loss_example}")
        print()

    #write disctionary to json


