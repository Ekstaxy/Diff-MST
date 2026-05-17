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
    def __init__(self, model, mix_console, sr=44100, clap_checkpoint=None, device='cuda'):
        self.model = model
        self.mix_console = mix_console
        self.sr = sr
        self.device = device

        print(f"[INFO] Initializing CLAP loss function with checkpoint: {clap_checkpoint}")
        self.clap_loss_fn = CLAPFeatureLoss(ckpt_path=clap_checkpoint)
        self.text_encoder = CLAPTextEncoder()

    def optimize(
        self, 
        mix: torch.Tensor, 
        raw_tracks: torch.Tensor, 
        processed_tracks: torch.Tensor, 
        prompt_str: str, 
        neg_str: str, 
        audio: torch.Tensor = None,
        target_track_idx=0, 
        track_start_idx=0, 
        length=882000, 
        num_steps=100, 
        lr=2e-4, 
        target_lufs_db=-14.0
    ):
        bs, chs, seq_len = raw_tracks.size()

        # Validate track_start_idx and length
        if track_start_idx + length > seq_len:
            raise ValueError(f"track_start_idx + length exceeds track length. Got track_start_idx={track_start_idx}, length={length}, but track length is {seq_len}.")

        # Extract the segment of the raw tracks to be mixed, ensuring it has the correct shape for the model
        mix_tracks = raw_tracks[..., track_start_idx : track_start_idx + length]

        # Validate target_track_idx
        if target_track_idx < -1 or target_track_idx >= chs:
            raise ValueError(f"Invalid target_track_idx {target_track_idx} for {chs} channels.")

        with torch.no_grad():
            # The initial embedding for the entire mix
            # The shape is (1, 2*num_tracks, emb_dim) because the model expects interleaved L/R channels for each track
            # The base is detached to ensure it doesn't receive gradients during optimization
            mono_processed_tracks = processed_tracks.mean(dim=1)
            print(f"[INFO] Mono processed tracks shape for embedding: {mono_processed_tracks.shape}")
            full_base_embedding = self.model.mix_encoder(mono_processed_tracks.clone()).detach()

        # Extract the target embedding and make it learnable
        initial_target_feature = full_base_embedding[:, target_track_idx : target_track_idx + 1, :]
        print(f"[INFO] Initial reference feature shape: {initial_target_feature.shape}")
        learnable_target_embedding = torch.nn.Parameter(initial_target_feature, requires_grad=True)
        print(f"[INFO] Learnable target embedding shape: {learnable_target_embedding.shape}")
        optimizer = torch.optim.RAdam([learnable_target_embedding], lr=lr)

        # The base embedding for the entire mix
        # This will be used as a fixed reference during optimization, and only the target track's embedding will be updated
        base_embedding = full_base_embedding.clone().detach()

        ito_embedding = base_embedding.clone()
        ito_embedding[0, target_track_idx, :] = learnable_target_embedding[0, 0, :]
        print(f"[INFO] Initial ITO embedding shape: {ito_embedding.shape}")

        with torch.no_grad():
            init_target_track_stereo = processed_tracks[:, :, target_track_idx, track_start_idx : track_start_idx + length]
            init_target_track_mono = init_target_track_stereo.mean(dim=1, keepdim=True)
            if audio is not None:
                init_audio_ref = audio[:, target_track_idx:target_track_idx+1, track_start_idx : track_start_idx + length] if audio.shape[-1] > length else audio[:, target_track_idx:target_track_idx+1, :]
                init_loss = self.clap_loss_fn(init_target_track_mono, init_audio_ref, sample_rate=self.sr, distance_fn="cosine", neg_target=None)
            else:
                init_loss = self.clap_loss_fn(init_target_track_mono, prompt_str, sample_rate=self.sr, distance_fn="cosine", neg_target=neg_str)
            print(f"[INFO] Initial Loss (Step 0 / Before ITO): {init_loss.item():.4f}")

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
                    ref_audio = ref_audio.view(1, 2*chs, -1)

                # print(f"[DEBUG] Step {step}: ref_audio shape: {ref_audio.shape}, ito_embedding shape: {ito_embedding.shape}")

                result = run_diffmst(
                    mix_tracks.clone(),
                    ref_audio.clone(),
                    self.model,
                    self.mix_console,
                    track_start_idx=track_start_idx,
                    ref_start_idx=track_start_idx,
                    ito_embedding=ito_embedding,
                    use_master_bus=False
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

                if audio is not None:
                    # audio 的 shape 是 (1, num_tracks, length)
                    target_audio_ref = audio[:, target_track_idx:target_track_idx+1, :]
                    clap_loss = self.clap_loss_fn(target_track_mono, target_audio_ref, sample_rate=self.sr, distance_fn="cosine", neg_target=None)
                else:
                    clap_loss = self.clap_loss_fn(target_track_mono, prompt_str, sample_rate=self.sr, distance_fn="cosine", neg_target=neg_str)
                
                clap_loss.backward()
                if learnable_target_embedding.grad is not None:
                    optimizer.step()
                else:
                    print(f"[WARNING] No gradients for learnable_target_embedding at step {step}. Skipping optimizer step.")

                # print(f"[INFO] Step {step}: CLAP loss: {clap_loss.item()}")

                if clap_loss.item() < min_loss:
                    min_loss = clap_loss.item()
                    min_loss_step = step
                    best_mix = mix.clone().detach()
                    best_stems = processed_tracks.clone().detach()

                # --- (已被註解掉) 在這裡不應該每回合重新計算 base_embedding ---
                # 如果每回合把自己剛剛預測的結果重新送進 mix_encoder，會產生嚴重飄移 (feedback loop)
                # current_mono_processed = processed_tracks.mean(dim=1)
                # base_embedding = self.model.mix_encoder(current_mono_processed).detach()
                # -----------------------------------------------------------

                ito_embedding = base_embedding.clone()
                ito_embedding[0, target_track_idx, :] = learnable_target_embedding[0, 0, :]
                
                mix_lufs_db = meter.integrated_loudness(
                    mix.clone().detach().squeeze(0).cpu().permute(1, 0).numpy()
                )
                lufs_delta_db = target_lufs_db - mix_lufs_db
                mix = mix * 10 ** (lufs_delta_db / 20)

                all_results.append({
                    'step': step + 1,
                    'loss': clap_loss.item(),
                    'stems': processed_tracks.detach().cpu(),
                    'params': pred_track_param_dict,
                })

        print(f"[INFO] Optimization completed. Minimum loss: {min_loss} at step {min_loss_step + 1}")
        
        return best_mix, best_stems, all_results, min_loss, min_loss_step
