import sys
sys.modules['flash_attn'] = None
sys.modules['torchao'] = None
original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
import torch
import numpy as np
import json
import pathlib  
import argparse
import torchaudio
import tqdm
import pyloudnorm as pyln

from mst.utils import load_diffmst, run_diffmst
from mst.loss import CLAPFeatureLoss
from mst.modules import CLAPTextEncoder
from mst.ito import ITOptimizer

sys.argv = original_argv
def parse_args():
    parser = argparse.ArgumentParser()

    # Config for model
    parser.add_argument('--config', type=str, required=True, help='Path to the naive.py config file.')

    # Checkpoints
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file.')
    parser.add_argument('--clap_checkpoint', type=str, required=True, help='Path to the CLAP checkpoint file.')

    # Tracks and reference settings
    parser.add_argument('--tracks_path', type=str, help='Path to the directory containing the tracks\' stems.')
    parser.add_argument('--tracks_start_idx', type=int, default=0, help='Starting index of the tracks to process.')
    parser.add_argument('--ref_start_idx', type=int, default=0, help='Starting index of the reference tracks to use for inference.')
    parser.add_argument('--length', type=int, default=44100*10, help='Length of the generated audio in samples.')
    parser.add_argument('--num_tracks', type=int, default=None, help='Number of tracks to process.')

    # Output settings
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated audio files.')
    parser.add_argument('--target_lufs', type=float, default=-14.0, help='Target LUFS level for loudness normalization.')

    # ITO settings
    parser.add_argument('--ito_iterations', type=int, default=50, help='Number of iterations for the ITO optimization process.')
    parser.add_argument('--ito_lr', type=float, default=2e-4, help='Learning rate for the ITO optimization process.')
    parser.add_argument('--prompt_str', type=str, default="A well-balanced mix with clear vocals and punchy drums.", help='Text prompt describing the desired mix characteristics for ITO optimization.')
    parser.add_argument('--neg_str', type=str, default="", help='Negative text prompt describing the undesired mix characteristics for ITO optimization.')
    parser.add_argument('--target_track_idx', type=int, default=0, help='Index of the track to optimize during ITO.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Load the model and CLAP text encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, mix_console = load_diffmst(
        config_path=args.config, 
        ckpt_path=args.checkpoint, 
        map_location=device
    )
    model.to(device)
    mix_console.to(device)
    clap_text_encoder = CLAPTextEncoder(args.clap_checkpoint).to(device)
    ito_optimizer = ITOptimizer(model, mix_console, clap_checkpoint=args.clap_checkpoint, device=device)

    # Tracks processing
    '''
    Assume my inference dataset is organized as follows:
    test_songs/
    ├── Song_A_Normal/
    │   ├── track_inst.wav
    │   └── track_vocal.wav
    |   └── ref_inst.wav
    |   └── ref_vocal.wav
    ├── Song_A_100Hz/
    │   ├── track_inst.wav
    │   └── track_vocal.wav
    |   └── ref_inst.wav
    |   └── ref_vocal.wav
    └── Song_B_Normal/
        ├── track_inst.wav
        └── track_vocal.wav
        └── ref_inst.wav
        └── ref_vocal.wav
    '''
    song_folders_path = pathlib.Path(args.tracks_path)
    
    # Safely get all song directories
    song_dirs = sorted([d for d in song_folders_path.iterdir() if d.is_dir()])[:args.num_tracks]

    # Prepare the output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meter = pyln.Meter(44100)
    target_lufs_db = args.target_lufs

    for song_dir in tqdm.tqdm(song_dirs, desc='Processing songs'):
        inst_path = song_dir / 'track_inst.wav'
        vocal_path = song_dir / 'track_vocal.wav'
        ref_inst_path = song_dir / 'ref_inst.wav'
        ref_vocal_path = song_dir / 'ref_vocal.wav'

        # Verify both stems exist before proceeding
        if not inst_path.exists() or not vocal_path.exists():
            print(f"Skipping {song_dir.name}: missing stems")
            continue

        # Verify reference audio exists
        if not ref_inst_path.exists() or not ref_vocal_path.exists():
            print(f"Skipping {song_dir.name}: missing reference audio")
            continue

        '''
        This part should be modified if in future I want to support more than 2 stems. 
        For now, I will just load the instrumental and vocal tracks and stack them along the channel dimension to create a 2-channel input for the model. 
        The track audio will also be loaded and stacked in the same way to create a 2-channel reference for the ITO optimization process
        '''
        # Load instrumental and vocal tracks and resample if necessary
        inst_audio, inst_sr = torchaudio.load(inst_path.as_posix(), backend='soundfile')
        vocal_audio, vocal_sr = torchaudio.load(vocal_path.as_posix(), backend='soundfile')
        if inst_sr != 44100:
            inst_audio = torchaudio.transforms.Resample(orig_freq=inst_sr, new_freq=44100)(inst_audio)
            inst_sr = 44100
        if vocal_sr != 44100:
            vocal_audio = torchaudio.transforms.Resample(orig_freq=vocal_sr, new_freq=44100)(vocal_audio)
            vocal_sr = 44100
        # Need consider every track has same length. 
        # If not, I will pad the shorter one with zeros to match the length of the longer one.
        max_length = max(inst_audio.shape[1], vocal_audio.shape[1])
        if inst_audio.shape[1] < max_length:
            padding = max_length - inst_audio.shape[1]
            inst_audio = torch.nn.functional.pad(inst_audio, (0, padding))
        if vocal_audio.shape[1] < max_length:
            padding = max_length - vocal_audio.shape[1]
            vocal_audio = torch.nn.functional.pad(vocal_audio, (0, padding))
        tracks = torch.cat([inst_audio, vocal_audio], dim=0)
        
        '''
        This part should be modified if in future I want to support more than 2 stems. 
        For now, I will just concatenate the instrumental and vocal tracks along the channel dimension to create a 2-channel input for the model. 
        The reference audio will also be concatenated in the same way to create a 2-channel reference for the ITO optimization process.
        '''
        # Load reference audio (instrumental and vocal) and resample if necessary
        ref_inst_audio, ref_sr = torchaudio.load(ref_inst_path.as_posix(), backend='soundfile')
        ref_vocal_audio, vocal_sr = torchaudio.load(ref_vocal_path.as_posix(), backend='soundfile')
        if ref_sr != 44100:
            ref_inst_audio = torchaudio.transforms.Resample(orig_freq=ref_sr, new_freq=44100)(ref_inst_audio)
            ref_sr = 44100
        if vocal_sr != 44100:
            ref_vocal_audio = torchaudio.transforms.Resample(orig_freq=vocal_sr, new_freq=44100)(ref_vocal_audio)
            vocal_sr = 44100
        # Make each of the reference track mono by averaging the channels if they are stereo
        if ref_inst_audio.shape[0] > 1:
            ref_inst_audio = ref_inst_audio.mean(dim=0, keepdim=True)
        if ref_vocal_audio.shape[0] > 1:
            ref_vocal_audio = ref_vocal_audio.mean(dim=0, keepdim=True)
        # Pad the reference audio to match the length of the input tracks
        max_length = max(ref_inst_audio.shape[1], ref_vocal_audio.shape[1])
        if ref_inst_audio.shape[1] < max_length:
            padding = max_length - ref_inst_audio.shape[1]
            ref_inst_audio = torch.nn.functional.pad(ref_inst_audio, (0, padding))
        if ref_vocal_audio.shape[1] < max_length:
            padding = max_length - ref_vocal_audio.shape[1]
            ref_vocal_audio = torch.nn.functional.pad(ref_vocal_audio, (0, padding))
        # ref_audio shape check dataloader.py with shape (2, length)
        ref_audio = torch.cat([ref_inst_audio, ref_vocal_audio], dim=0)

        # Handle the length of the input tracks and reference audio.
        if args.tracks_start_idx + args.length > tracks.shape[-1]:
                print(f"[Warning] Tracks too short for this section.")
        if args.ref_start_idx + args.length > ref_audio.shape[-1]:
            print(f"[Warning] Reference too short for this section.")

        mix_tracks = tracks[:, args.tracks_start_idx:args.tracks_start_idx + args.length]
        ref_audio = ref_audio[:, args.ref_start_idx:args.ref_start_idx + args.length]

        with torch.no_grad():
            result = run_diffmst(
                mix_tracks.clone().unsqueeze(0).to(device),     # Shape: (1, 2, length)
                ref_audio.clone().unsqueeze(0).to(device),      # Shape: (1, 2, length)
                model,
                mix_console,
                track_start_idx=args.tracks_start_idx,
                ref_start_idx=args.ref_start_idx,
                use_master_bus=False
            )
            (
                pred_mix,
                pred_mixed_tracks,
                pred_track_param_dict,
                pred_fx_bus_param_dict,
                pred_master_bus_param_dict,
            ) = result

        # The predicted mix and mixed tracks are expected to have shape (1, 2, length) and (1, 2, num_tracks, length) respectively.
        bs, chs, length = pred_mix.shape

        song_output_dir = output_dir / song_dir.name
        song_output_dir.mkdir(parents=True, exist_ok=True)

        mix_lufs_db = meter.integrated_loudness(
            pred_mix.squeeze(0).cpu().permute(1, 0).numpy()
        )
        lufs_delta_db = target_lufs_db - mix_lufs_db
        pred_mix = pred_mix * 10 ** (lufs_delta_db / 20)
        pred_mixed_tracks = pred_mixed_tracks * 10 ** (lufs_delta_db / 20)

        mix_filepath = song_output_dir / "pred_mix.wav"
        torchaudio.save(mix_filepath, pred_mix.view(chs, -1).cpu(), 44100)

        for track_idx in range(pred_mixed_tracks.shape[1]):
            track_filepath = song_output_dir / f"pred_track_{track_idx}.wav"
            torchaudio.save(track_filepath, pred_mixed_tracks.clone().squeeze(0)[:, track_idx, :].cpu(), 44100)

        # ITO optimization process
        if args.ito_iterations > 0:
            best_mix, best_stems, all_results, min_loss, min_loss_step = ito_optimizer.optimize(
                mix = pred_mix.clone().to(device),                          # Shape: (1, 2, length)
                raw_tracks = mix_tracks.clone().unsqueeze(0).to(device),    # Shape: (1, 2, length)
                processed_tracks = pred_mixed_tracks.clone().to(device),    # Shape: (1, 2, num_tracks, length)
                prompt_str = args.prompt_str,
                neg_str = args.neg_str,
                target_track_idx = args.target_track_idx,
                track_start_idx = 0,
                length = args.length,
                num_steps = args.ito_iterations,
                lr = args.ito_lr
            )

            mix_lufs_db = meter.integrated_loudness(
                best_mix.squeeze(0).cpu().permute(1, 0).numpy()
            )
            lufs_delta_db = target_lufs_db - mix_lufs_db
            best_mix = best_mix * 10 ** (lufs_delta_db / 20)
            best_stems = best_stems * 10 ** (lufs_delta_db / 20)

            # Save the best mix from ITO optimization
            best_mix_filepath = song_output_dir / "ito_optimized_mix.wav"
            torchaudio.save(best_mix_filepath, best_mix.view(chs, -1).cpu(), 44100)

            # Save the best stems from ITO optimization
            for track_idx in range(best_stems.shape[1]):
                track_filepath = song_output_dir / f"ito_optimized_track_{track_idx}.wav"
                torchaudio.save(track_filepath, best_stems.squeeze(0)[:, track_idx, :].cpu(), 44100)

if __name__ == "__main__":
    print("====== Inference Script Started ======")
    main()