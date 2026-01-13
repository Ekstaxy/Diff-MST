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

def parse_args():
    parser = argparse.ArgumentParser(description='Generate audio examples for listening test')
    # Model configs
    parser.add_argument("--config", type=str, required=True,
                        help='Path to naive.yaml')
    parser.add_argument("--checkpoint", type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument("--clap_checkpoint", type=str, default=None,
                        help='Path to CLAP model checkpoint')
    
    # Audio parameters
    parser.add_argument("--tracks_path", type=str, 
                        default="/mnt/gestalt/home/rakec/data/diff-mst/MedleyDB_V2/V2/TleilaxEnsemble_Late/TleilaxEnsemble_Late_RAW",
                        help='Path to folder with audio tracks to use for mixing')
    
    # Looping parameters
    parser.add_argument("--control_type", type=str, nargs='+', default=["audio"],
                        help="Control types to use for mixing (audio or text)")    

    parser.add_argument("--control_info", type=str, nargs='+', 
                    default=["/content/V2/TleilaxEnsemble_Late/TleilaxEnsemble_Late_MIX.wav", (1, 1, "Bright")],
                    help="Control information (file paths for audio, text prompts for text in format: (track, weight, 'text'). If track is -1, use master bus.)")
    
    # Verse/Chorus indices
    parser.add_argument('--track-verse-idx', type=int, required=True,
                        help='Track verse start index (samples)')
    parser.add_argument('--track-chorus-idx', type=int, required=True,
                        help='Track chorus start index (samples)')
    parser.add_argument('--ref-verse-idx', type=int, required=True,
                        help='Reference verse start index (samples)')
    parser.add_argument('--ref-chorus-idx', type=int, required=True,
                        help='Reference chorus start index (samples)')
    
    # ITO-Master Parameters
    parser.add_argument("--ito_num_step", type=int, default=4,
                        help='Number of ITO-Master steps to use during inference')
    
    # Other parameters
    parser.add_argument("--output_dir", type=str, default="./eval_outputs",
                        help='Directory to save generated audio examples')
    parser.add_argument("--exp_name", type=str, default="test", 
                        help='Experiment name for output folder')
    parser.add_argument('--target_lufs', type=float, default=-22.0,
                        help='Target output LUFS')

    parser.add_argument('--sum_only', type=bool, default=False,
                        help='Whether to only run the sum baseline')

    return parser.parse_args()

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

def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (torch.Tensor, np.ndarray)):
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.tolist()
    elif isinstance(obj, (float, int, str)):
        return obj
    else:
        return str(obj)
    
def audio_inference():
    pass

def main():
    args = parse_args()
    print(args.control_info)

    meter = pyln.Meter(44100)
    target_lufs_db = args.target_lufs
    output_dir = pathlib.Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = {
        "diffmst":{
            "model": load_diffmst(
                config_path=args.config,
                ckpt_path=args.checkpoint
            ),
            "func": run_diffmst
        },
        "sum": {
            "model": (None, None),
            "func": equal_loudness_mix,
        }
    }

    # Find all tracks
    track_filepaths = list(pathlib.Path(args.tracks_path).glob("*.wav"))
    print(f"[INFO] Found {len(track_filepaths)} tracks in {args.tracks_path}")

    tracks = []
    lengths = []
    for track_idx, track_filepath in enumerate(track_filepaths):
        audio, sr = torchaudio.load(track_filepath, backend="soundfile")

        if sr != 44100:
            audio = torchaudio.functional.resample(audio, sr, 44100)

        if audio.shape[0] == 2:
            audio = audio.mean(dim=0, keepdim=True)

        chs, seq_len = audio.shape

        for ch_idx in range(chs):
            tracks.append(audio[ch_idx : ch_idx + 1, :])
            lengths.append(audio.shape[-1])

    # Find max length and pad if shorter
    max_length = max(lengths)
    for track_idx in range(len(tracks)):
        tracks[track_idx] = torch.nn.functional.pad(
            tracks[track_idx], (0, max_length - lengths[track_idx])
        )
    
    tracks = torch.cat(tracks, dim=0)
    tracks = tracks.view(1, -1, max_length)

    print(f"[INFO] Tracks shape: {tracks.shape}")

    if args.sum_only:
        for song_section in ["verse"]:
            print(f"[INFO] Mixing {song_section} with sum baseline...")
            if song_section == "verse":
                track_start_idx = args.track_verse_idx
            else:
                track_start_idx = args.track_chorus_idx

            mix_tracks = tracks[..., track_start_idx : track_start_idx + (44100 * 10)]

            sum_mix, _, _, _ = equal_loudness_mix(mix_tracks)

            mix_lufs_db = meter.integrated_loudness(
                sum_mix.squeeze(0).permute(1, 0).numpy()
            )
            lufs_delta_db = target_lufs_db - mix_lufs_db
            sum_mix = sum_mix * 10 ** (lufs_delta_db / 20)

            mix_filepath = output_dir / f"sum-baseline-{song_section}-lufs-{int(target_lufs_db)}.wav"
            torchaudio.save(mix_filepath, sum_mix.view(2, -1), 44100)
        # return
        
    # Control type and info checks
    for c_idx, c_type in enumerate(args.control_type):
        assert c_type in ["audio", "text"], f"Unsupported control type: {c_type}"

        # Audio Control
        if c_type == "audio":
            example = {
                "tracks": args.tracks_path,
                "track_verse_start_idx": args.track_verse_idx,
                "track_chorus_start_idx": args.track_chorus_idx,
                "ref": args.control_info[c_idx],
                "ref_verse_start_idx": args.ref_verse_idx,
                "ref_chorus_start_idx": args.ref_chorus_idx
            }

            ref_audio, ref_sr = torchaudio.load(example["ref"], backend="soundfile")
            if ref_sr != 44100:
                ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 44100)
            ref_audio = ref_audio.view(1, 2, -1)
            print(f"[INFO] reference audio shape: {ref_audio.shape}")

            # Mix with audio reference
            print(f"[INFO] Mixing with Audio Reference...")

            track_start_idx = example["track_verse_start_idx"]
            ref_start_idx = example["ref_verse_start_idx"]

            if track_start_idx + 44100 * 10 * 2 > tracks.shape[-1]:
                print(f"[Warning] Tracks too short for this section.")
            if ref_start_idx + 44100 * 10 > ref_audio.shape[-1]:
                print(f"[Warning] Reference too short for this section.")

            mix_tracks = tracks
            mix_tracks = tracks[..., track_start_idx : track_start_idx + (44100 * 10 * 2)]
            track_start_idx = 0

            method_name = "diffmst"
            method = methods[method_name]
            print(f"[INFO] Applying method: {method_name}")

            model, mix_console = method["model"]
            model = model.to("cpu") if model is not None else None
            mix_console = mix_console.to("cpu") if mix_console is not None else None
            func = method["func"]

            with torch.no_grad():
                result = func(
                    mix_tracks.clone(),
                    ref_audio.clone(),
                    model,
                    mix_console,
                    track_start_idx=track_start_idx,
                    ref_start_idx=ref_start_idx,
                )

                (
                    pred_mix,
                    pred_mixed_tracks,
                    pred_track_param_dict,
                    pred_fx_bus_param_dict,
                    pred_master_bus_param_dict,
                ) = result

                bs, chs, seq_len = pred_mix.shape

                mix_lufs_db = meter.integrated_loudness(
                    pred_mix.squeeze(0).permute(1, 0).numpy()
                )
                lufs_delta_db = target_lufs_db - mix_lufs_db
                pred_mix = pred_mix * 10 ** (lufs_delta_db / 20)

                mix_filepath = output_dir / f"step{c_idx}-{method_name}-ref={song_section}.wav"
                torchaudio.save(mix_filepath, pred_mix.view(chs, -1), 44100)

                # Save individual processed stems
                stems_dir = output_dir / f"step{c_idx}-{method_name}-ref={song_section}-stems"
                stems_dir.mkdir(exist_ok=True)
                
                # Assuming batch size is 1
                print(pred_mixed_tracks.shape)
                num_tracks = pred_mixed_tracks.shape[2]
                for t_idx in range(num_tracks):
                    stem_audio = pred_mixed_tracks[0, :, t_idx, :]
                    stem_filename = f"track_{t_idx}.wav"
                    torchaudio.save(stems_dir / stem_filename, stem_audio, 44100)
                    
        # # Text Control
        # elif c_type == "text":
        #     text = args.control_info[c_idx]
        #     print(f"[INFO] Using text prompt: {text[2]}, weight: {text[1]}, track: {text[0]}")
            
        #     example = {
        #         "tracks": args.tracks_path,
        #         "track_verse_start_idx": args.track_verse_idx,
        #         "track_chorus_start_idx": args.track_chorus_idx,
        #         "ref": args.control_info[c_idx],
        #         "ref_verse_start_idx": args.ref_verse_idx,
        #         "ref_chorus_start_idx": args.ref_chorus_idx
        #     }

        #     num_tracks = pred_mixed_tracks.shape[2]     # pred_mixed_tracks: (bs, 2, num_tracks, seq_len)
        #     if example["ref"][0] < -1 or example["ref"][0] >= num_tracks:
        #         raise ValueError(f"Invalid track index {example['ref'][0]} for {num_tracks} tracks.")

        #     if example["ref"][0] == -1:
        #         ref_audio = pred_mix
        #     else:
        #         ref_audio = pred_mixed_tracks
        #         ref_audio = ref_audio.view(1, 2*num_tracks, -1)

        #     print(f"[INFO] reference audio shape: {ref_audio.shape}")
            
        #     prev_fx_bus_param_dict = pred_fx_bus_param_dict
        #     prev_track_param_dict = pred_track_param_dict
        #     prev_master_bus_param_dict = pred_master_bus_param_dict

        #     for song_section in ["verse"]:
        #         print(f"[INFO] Mixing {song_section}...")
        #         if song_section == "verse":
        #             track_start_idx = example["track_verse_start_idx"]
        #             ref_start_idx = example["ref_verse_start_idx"]
        #         else:
        #             track_start_idx = example["track_chorus_start_idx"]
        #             ref_start_idx = example["ref_chorus_start_idx"]

        #         if track_start_idx + 44100 * 10 > tracks.shape[-1]:
        #             print(f"[Warning] Tracks too short for this section.")
        #         if ref_start_idx + 44100 * 10 > ref_audio.shape[-1]:
        #             print(f"[Warning] Reference too short for this section.")


        #         method_name = "diffmst"
        #         method = methods[method_name]
        #         print(f"[INFO] Applying method: {method_name}")


        #         model, mix_console = method["model"]
        #         model = model.to("cpu") if model is not None else None
        #         mix_console = mix_console.to("cpu") if mix_console is not None else None
        #         func = method["func"]
                

        #         with torch.no_grad():
        #             result = func(
        #                 mix_tracks.clone(),
        #                 ref_audio.clone(),
        #                 model,
        #                 mix_console,
        #                 text=example["ref"],
        #                 interpolation="linear",
        #                 track_start_idx=track_start_idx,
        #                 ref_start_idx=ref_start_idx,
        #                 prev_fx_bus_param_dict=prev_fx_bus_param_dict,
        #                 prev_master_bus_param_dict=prev_master_bus_param_dict,
        #                 prev_track_param_dict=prev_track_param_dict,
        #             )

        #             (
        #                 pred_mix,
        #                 pred_mixed_tracks,
        #                 pred_track_param_dict,
        #                 pred_fx_bus_param_dict,
        #                 pred_master_bus_param_dict,
        #             ) = result
                    
                    
                    
        #             bs, chs, seq_len = pred_mix.shape

        #             mix_lufs_db = meter.integrated_loudness(
        #                 pred_mix.squeeze(0).permute(1, 0).numpy()
        #             )
        #             lufs_delta_db = target_lufs_db - mix_lufs_db
        #             pred_mix = pred_mix * 10 ** (lufs_delta_db / 20)

        #             mix_filepath = output_dir / f"step{c_idx}-{method_name}-ref={song_section}.wav"
        #             torchaudio.save(mix_filepath, pred_mix.view(chs, -1), 44100)

        #             # Save individual processed stems
        #             stems_dir = output_dir / f"step{c_idx}-{method_name}-ref={song_section}-stems"
        #             stems_dir.mkdir(exist_ok=True)
                    
        #             # Assuming batch size is 1
        #             num_tracks = pred_mixed_tracks.shape[2]
        #             for t_idx in range(num_tracks):
        #                 stem_audio = pred_mixed_tracks[0, :, t_idx, :]
        #                 stem_filename = f"track_{t_idx}.wav"
        #                 torchaudio.save(stems_dir / stem_filename, stem_audio, 44100)

        #             json_data = {
        #                 "step": c_idx,
        #                 "type": "text",
        #                 "prompt": text[2],
        #                 "weight": text[1],
        #                 "target_track": text[0],
        #                 "section": song_section,
        #                 "prev_param": {
        #                     "tracks": make_serializable(prev_track_param_dict),
        #                     "fx_bus": make_serializable(prev_fx_bus_param_dict),
        #                     "master_bus": make_serializable(prev_master_bus_param_dict)
        #                 },
        #                 "pred_param": {
        #                     "tracks": make_serializable(pred_track_param_dict),
        #                     "fx_bus": make_serializable(pred_fx_bus_param_dict),
        #                     "master_bus": make_serializable(pred_master_bus_param_dict)
        #                 }
        #             }
        #             json_path = output_dir / f"step{c_idx}-{method_name}-ref={song_section}.json"
        #             with open(json_path, 'w') as f:
        #                 json.dump(json_data, f, indent=4)

        # ITO
        text_info = args.control_info[1]
        prompt_str = text_info[2]
        print(text_info[0], type(text_info[0]))
        track_idx = int(text_info[0])
        bs, num_tracks, seq_len = tracks.size()
        print(f"[INFO] Using ITO text prompt: {prompt_str}")

        clap_loss_fn = CLAPFeatureLoss(ckpt_path=args.clap_checkpoint)

        if track_idx >= 0:

            with torch.no_grad():

                full_base_embedding = model.mix_encoder(pred_mixed_tracks)
                
                # 確保基底不需要梯度
                full_base_embedding = full_base_embedding.detach()
                print(f"[INFO] pred_mixed_tracks shape: {pred_mixed_tracks.shape}")
                
            
            num_tracks_mix = full_base_embedding.size(1) // 2

            # target_L = full_base_embedding[0, track_idx, :]
            # target_R = full_base_embedding[0, track_idx + num_tracks_mix, :]
            target_L = full_base_embedding[0:1, track_idx : track_idx + 1, :]
            target_R = full_base_embedding[0:1, track_idx + num_tracks_mix : track_idx + num_tracks_mix + 1, :]
            
            initial_reference_feature = torch.cat([target_L, target_R], dim=1) 
            print(f"[INFO] Initial reference feature shape: {initial_reference_feature.shape}")

        else:
            raise ValueError("For single track ITO, target_track_idx must be >= 0")
            
        fit_embedding = torch.nn.Parameter(initial_reference_feature, requires_grad=True)
        print(f"[INFO] Fitting embedding shape: {fit_embedding.shape}")
        optimizer = torch.optim.AdamW([fit_embedding], lr=1e-2)

        text_encoder = CLAPTextEncoder()
        ito_embedding = full_base_embedding.clone()  
        ito_embedding[0, track_idx, :] = fit_embedding[0, 0, :]
        ito_embedding[0, track_idx + num_tracks_mix, :] = fit_embedding[0, 1, :]
        print(f"[INFO] Initial ITO embedding shape: {ito_embedding.shape}")
        
        min_loss = float('inf')
        min_loss_step = 0
        all_results = []

        for ito_step in tqdm.tqdm(range(args.ito_num_step)):
            print(f"[INFO] ITO step {ito_step+1}/{args.ito_num_step}...")
            optimizer.zero_grad()
    
            example = {
                "tracks": args.tracks_path,
                "track_verse_start_idx": args.track_verse_idx,
                "track_chorus_start_idx": args.track_chorus_idx,
                "ref": args.control_info[1],
                "ref_verse_start_idx": args.ref_verse_idx,
                "ref_chorus_start_idx": args.ref_chorus_idx
            }

            num_tracks = pred_mixed_tracks.shape[2]     # pred_mixed_tracks: (bs, 2, num_tracks, seq_len)

            if example["ref"][0] < -1 or example["ref"][0] >= num_tracks:
                raise ValueError(f"Invalid track index {example['ref'][0]} for {num_tracks} tracks.")

            if example["ref"][0] == -1:
                ref_audio = pred_mix.detach()
            else:
                ref_audio = pred_mixed_tracks.detach()
                ref_audio = ref_audio.view(1, 2*num_tracks, -1)

            print(f"[INFO] reference audio shape: {ref_audio.shape}")
            
            prev_fx_bus_param_dict = pred_fx_bus_param_dict
            prev_track_param_dict = pred_track_param_dict
            prev_master_bus_param_dict = pred_master_bus_param_dict

            # Mix with ITO Reference
            print(f"[INFO] Mixing with ITO Reference...")

            track_start_idx = example["track_verse_start_idx"]
            ref_start_idx = example["ref_verse_start_idx"]

            if track_start_idx + 44100 * 10 * 2 > tracks.shape[-1]:
                print(f"[Warning] Tracks too short for this section.")
            if ref_start_idx + 44100 * 10 > ref_audio.shape[-1]:
                print(f"[Warning] Reference too short for this section.")

            mix_tracks = tracks
            mix_tracks = tracks[..., track_start_idx : track_start_idx + (44100 * 10 * 2)]
            track_start_idx = 0

            method_name = "diffmst"
            method = methods[method_name]
            print(f"[INFO] Applying method: {method_name}")

            model, mix_console = method["model"]
            model = model.to("cpu") if model is not None else None
            mix_console = mix_console.to("cpu") if mix_console is not None else None
            func = method["func"]

            result = func(
                mix_tracks.clone(),
                ref_audio.clone(),
                model,
                mix_console,
                track_start_idx=track_start_idx,
                ref_start_idx=ref_start_idx,
                ito_embedding = ito_embedding,
            )

            (
                pred_mix,
                pred_mixed_tracks,
                pred_track_param_dict,
                pred_fx_bus_param_dict,
                pred_master_bus_param_dict,
            ) = result

            bs, chs, seq_len = pred_mix.shape

            # Select the target track audio and convert to mono
            target_track_audio = pred_mixed_tracks[:, :, track_idx, :]
            target_track_mono = target_track_audio.mean(dim=1, keepdim=True)

            total_clap_loss = clap_loss_fn(target_track_mono, prompt_str, sample_rate=44100, distance_fn="cosine")
            total_clap_loss.backward()
            optimizer.step()

            if total_clap_loss < min_loss:
                min_loss = total_clap_loss.item()
                min_loss_step = ito_step

            current_embeddings = model.mix_encoder(pred_mixed_tracks.clone())
            
            ito_embedding = current_embeddings.detach()
            ito_embedding[0, track_idx, :] = fit_embedding[0, 0, :]
            ito_embedding[0, track_idx + num_tracks_mix, :] = fit_embedding[0, 1, :]

            mix_lufs_db = meter.integrated_loudness(
                pred_mix.clone().detach().squeeze(0).permute(1, 0).numpy()
            )
            lufs_delta_db = target_lufs_db - mix_lufs_db
            pred_mix = pred_mix * 10 ** (lufs_delta_db / 20)

            all_results.append({
                'step': ito_step + 1,
                'loss': total_clap_loss.item(),
                'audio': pred_mix.cpu(),
                'stems': pred_mixed_tracks.cpu(),
                'params': pred_track_param_dict,
            })

            # Intermediate saving logic removed to save only best result

        # Save Best Result
        print(f"[INFO] Best step: {all_results[min_loss_step]['step']}, Loss: {min_loss}")
        best_result = all_results[min_loss_step]
        best_mix = best_result['audio']
        best_stems = best_result['stems']
        bs, chs, seq_len = best_mix.shape
        
        mix_filepath = output_dir / f"step{c_idx}-{method_name}-ref={song_section}-best.wav"
        torchaudio.save(mix_filepath, best_mix.view(chs, -1), 44100)
        
        # Save individual processed stems for the BEST result only
        stems_dir = output_dir / f"step{c_idx}-{method_name}-ref={song_section}-stems-best"
        stems_dir.mkdir(exist_ok=True)
            
        print(best_stems.shape)
        num_tracks = best_stems.shape[2]
        for t_idx in range(num_tracks):
            stem_audio = best_stems[0, :, t_idx, :]
            stem_filename = f"track_{t_idx}.wav"
            torchaudio.save(stems_dir / stem_filename, stem_audio, 44100)

if __name__ == "__main__":
    main()