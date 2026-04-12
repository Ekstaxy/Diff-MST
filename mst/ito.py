import torch
import torchaudio
import pathlib
import argparse
import librosa
import numpy as np
import tqdm
import pyloudnorm as pyln

from mst.utils import load_diffmst, run_diffmst
from mst.loss import CLAPFeatureLoss
from mst.modules import CLAPTextEncoder

class ITOptimizer:
    def __init__(self, model, mix_console, sr=44100, clap_checkpoint=None):
        self.model = model
        self.mix_console = mix_console
        self.sr = sr

        print(f"[INFO] Initializing CLAP loss function with checkpoint: {clap_checkpoint}")
        self.clap_loss_fn = CLAPFeatureLoss(ckpt_path=clap_checkpoint)
        self.text_encoder = CLAPTextEncoder()

    def optimize(self, mix, raw_tracks, processed_tracks, prompt_str, neg_str, target_track_idx=0, track_start_idx=0, length=882000, num_steps=100, lr=1e-3, target_lufs_db=-14.0):
        bs, num_tracks, seq_len = raw_tracks.size()

        # Validate track_start_idx and length
        if track_start_idx + length > seq_len:
            raise ValueError(f"track_start_idx + length exceeds track length. Got track_start_idx={track_start_idx}, length={length}, but track length is {seq_len}.")

        # Extract the segment of the raw tracks to be mixed, ensuring it has the correct shape for the model
        mix_tracks = raw_tracks[..., track_start_idx : track_start_idx + length]

        # Validate target_track_idx
        if target_track_idx < -1 or target_track_idx >= num_tracks:
            raise ValueError(f"Invalid target_track_idx {target_track_idx} for {num_tracks} tracks.")

        with torch.no_grad():

            # The initial embedding for the entire mix
            # The shape is (1, 2*num_tracks, emb_dim) because the model expects interleaved L/R channels for each track
            # The base is detached to ensure it doesn't receive gradients during optimization
            full_base_embedding = self.model.mix_encoder(processed_tracks.clone().view(1, 2*num_tracks, -1))
            full_base_embedding = full_base_embedding.detach()

        num_tracks_mix = full_base_embedding.size(1) // 2

        # Extract the target embedding and make it learnable
        target_L = full_base_embedding[0:1, target_track_idx : target_track_idx + 1, :]
        target_R = full_base_embedding[0:1, target_track_idx + num_tracks_mix : target_track_idx + num_tracks_mix + 1, :]
        initial_target_feature = torch.cat([target_L, target_R], dim=1)
        print(f"[INFO] Initial reference feature shape: {initial_target_feature.shape}")
        learnable_target_embedding = torch.nn.Parameter(initial_target_feature, requires_grad=True)
        print(f"[INFO] Learnable target embedding shape: {learnable_target_embedding.shape}")
        optimizer = torch.optim.RAdam([learnable_target_embedding], lr=lr)

        # The base embedding for the entire mix
        # This will be used as a fixed reference during optimization, and only the target track's embedding will be updated
        base_embedding = full_base_embedding.clone().detach()

        # Create a mask to isolate the target track's embedding
        mask = torch.zeros_like(base_embedding)
        mask[0, target_track_idx, :] = 1.0
        mask[0, target_track_idx + num_tracks_mix, :] = 1.0

        # The learnable target embedding is expanded to the full mix embedding shape
        # This ensures that during optimization, only the target track's embedding is updated while the rest of the mix embedding remains fixed
        learnable_target_embedding_expanded = torch.zeros_like(base_embedding)
        learnable_target_embedding_expanded[0, target_track_idx, :] = learnable_target_embedding[0, 0, :]
        learnable_target_embedding_expanded[0, target_track_idx + num_tracks_mix, :] = learnable_target_embedding[0, 1, :]

        ito_embedding = (learnable_target_embedding_expanded * mask) + (base_embedding * (1 - mask))
        print(f"[INFO] Initial ITO embedding shape: {ito_embedding.shape}")

        min_loss = float('inf')
        min_loss_step = 0
        best_mix = None
        best_stems = None
        all_results = []

        meter = pyln.Meter(self.sr)

        # Optimization loop
        with torch.enable_grad():
            for step in tqdm.tqdm(range(num_steps)):
                optimizer.zero_grad()

                if target_track_idx == -1:
                    ref_audio = mix.detach()
                else:
                    ref_audio = processed_tracks.detach()
                    ref_audio = ref_audio.view(1, 2*num_tracks, -1)

                print(f"[DEBUG] Step {step}: ref_audio shape: {ref_audio.shape}, ito_embedding shape: {ito_embedding.shape}")

                result = run_diffmst(
                    mix_tracks.clone(),
                    ref_audio.clone(),
                    self.model,
                    self.mix_console,
                    track_start_idx=track_start_idx,
                    ref_start_idx=track_start_idx,
                    ito_embedding=ito_embedding
                )
                (
                    mix,
                    processed_tracks,
                    pred_track_param_dict,
                    pred_fx_bus_param_dict,
                    pred_master_bus_param_dict,
                ) = result

                bs, chs, seq_len = mix.shape

                target_track_stereo = processed_tracks[:, :, target_track_idx, :]
                target_track_mono = target_track_stereo.mean(dim=1, keepdim=True)

                clap_loss = self.clap_loss_fn(target_track_mono, prompt_str, sample_rate=self.sr, distance_fn="cosine", neg_target=neg_str)
                clap_loss.backward()
                if learnable_target_embedding.grad is not None:
                    optimizer.step()
                else:
                    print(f"[WARNING] No gradients for learnable_target_embedding at step {step}. Skipping optimizer step.")

                print(f"[INFO] Step {step}: CLAP loss: {clap_loss.item()}")

                if clap_loss.item() < min_loss:
                    min_loss = clap_loss.item()
                    min_loss_step = step
                    best_mix = mix.clone().detach()
                    best_stems = processed_tracks.clone().detach()

                # Prepare the ITO embedding for the next iteration
                current_embedings = self.model.mix_encoder(processed_tracks.clone().view(1, 2*num_tracks, -1)).detach()

                # The base embedding for the entire mix
                # This will be used as a fixed reference during optimization, and only the target track's embedding will be updated
                base_embedding = current_embedings.detach()

                # Create a mask to isolate the target track's embedding
                mask = torch.zeros_like(base_embedding)
                mask[0, target_track_idx, :] = 1.0
                mask[0, target_track_idx + num_tracks_mix, :] = 1.0

                # The learnable target embedding is expanded to the full mix embedding shape
                # This ensures that during optimization, only the target track's embedding is updated while the rest of the mix embedding remains fixed
                learnable_target_embedding_expanded = torch.zeros_like(base_embedding)
                learnable_target_embedding_expanded[0, target_track_idx, :] = learnable_target_embedding[0, 0, :]
                learnable_target_embedding_expanded[0, target_track_idx + num_tracks_mix, :] = learnable_target_embedding[0, 1, :]

                ito_embedding = (learnable_target_embedding_expanded * mask) + (base_embedding * (1 - mask))
                
                mix_lufs_db = meter.integrated_loudness(
                    mix.clone().detach().squeeze(0).permute(1, 0).numpy()
                )
                lufs_delta_db = target_lufs_db - mix_lufs_db
                mix = mix * 10 ** (lufs_delta_db / 20)

                all_results.append({
                    'step': step + 1,
                    'loss': clap_loss.item(),
                    'stems': processed_tracks.cpu(),
                    'params': pred_track_param_dict,
                })

        print(f"[INFO] Optimization completed. Minimum loss: {min_loss} at step {min_loss_step + 1}")
        
        return best_mix, best_stems, all_results, min_loss, min_loss_step
