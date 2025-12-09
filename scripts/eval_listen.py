# run pretrained models over evaluation set to generate audio examples for the listening test
import os
import torch
import torchaudio
import pyloudnorm as pyln
import laion_clap
from mst.utils import load_diffmst, run_diffmst
import numpy as np


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

    return sum_mix, None, None, None, None


if __name__ == "__main__":
    meter = pyln.Meter(44100)
    target_lufs_db = -22.0
    output_dir = "outputs/listen_1"
    use_text_optimize = True
    os.makedirs(output_dir, exist_ok=True)
    clap_model = laion_clap.CLAP_Module(enable_fusion=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    optimize_option = "slerp"
    # optimize_option = "ADAM"

    methods = {
        "diffmst-16": {
            "model": load_diffmst(
                "/import/c4dm-datasets-ext/Diff-MST/DiffMST/b4naquji/config.yaml",
                "/import/c4dm-datasets-ext/Diff-MST/DiffMST/b4naquji/checkpoints/epoch=191-step=626608.ckpt",
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
        "ecstasy": {
            "tracks": "/import/c4dm-datasets-ext/diffmst-examples/song1/BenFlowers_Ecstasy_Full/",
            "track_verse_start_idx": 1190700,
            "track_chorus_start_idx": 2381400,
            "ref": "/import/c4dm-datasets-ext/diffmst-examples/song1/ref/_Feel it all Around_ by Washed Out (Portlandia Theme)_01.wav",
            "ref_verse_start_idx": 970200,
            "ref_chorus_start_idx": 198450,
        },
        "by-my-side": {
            "tracks": "/import/c4dm-datasets-ext/diffmst-examples/song2/Kat Wright_By My Side/",
            "track_verse_start_idx": 1146600,
            "track_chorus_start_idx": 7144200,
            "ref": "/import/c4dm-datasets-ext/diffmst-examples/song2/ref/The Dip - Paddle To The Stars (Lyric Video)_01.wav",
            "ref_verse_start_idx": 661500,
            "ref_chorus_start_idx": 2028600,
        },
        "haunted-aged": {
            "tracks": "/import/c4dm-datasets-ext/diffmst-examples/song3/Titanium_HauntedAge_Full/",
            "track_verse_start_idx": 1675800,
            "track_chorus_start_idx": 3439800,
            "ref": "/import/c4dm-datasets-ext/diffmst-examples/song3/ref/Architects - _Doomsday__01.wav",
            "ref_verse_start_idx": 4630500,
            "ref_chorus_start_idx": 6570900,
        },
    }

    for example_name, example in examples.items():
        print(example_name)
        example_dir = os.path.join(output_dir, example_name)
        os.makedirs(example_dir, exist_ok=True)
        # load reference mix
        ref_audio, ref_sr = torchaudio.load(example["ref"], backend="soundfile")
        if ref_sr != 44100:
            ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 44100)
        print(ref_audio.shape, ref_sr)

        # first find all the tracks
        track_filepaths = []
        for root, dirs, files in os.walk(example["tracks"]):
            for filepath in files:
                if filepath.endswith(".wav"):
                    track_filepaths.append(os.path.join(root, filepath))

        print(f"Found {len(track_filepaths)} tracks.")

        # load the tracks
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

            #print(track_idx, os.path.basename(track_filepath), audio.shape, sr, lufs_db)

            if audio.shape[0] == 2:
                audio = audio.mean(dim=0, keepdim=True)

            chs, seq_len = audio.shape

            for ch_idx in range(chs):
                tracks.append(audio[ch_idx : ch_idx + 1, :])
                lengths.append(audio.shape[-1])
        print("Loaded tracks.")
        # find max length and pad if shorter
        max_length = max(lengths)
        for track_idx in range(len(tracks)):
            tracks[track_idx] = torch.nn.functional.pad(
                tracks[track_idx], (0, max_length - lengths[track_idx])
            )
        print("Padded tracks.")
        # stack into a tensor
        tracks = torch.cat(tracks, dim=0)
        tracks = tracks.view(1, -1, max_length)
        ref_audio = ref_audio.view(1, 2, -1)

        # crop tracks to max of 60 seconds or so
        # tracks = tracks[..., :4194304]

        print(tracks.shape)

        # create a sum mix with equal loudness
        sum_mix = torch.sum(tracks, dim=1, keepdim=True).squeeze(0)
        sum_filepath = os.path.join(example_dir, f"{example_name}-sum.wav")
        os.makepath(sum_filepath)
        print("sum_mix path created")

        # loudness normalize the sum mix
        sum_lufs_db = meter.integrated_loudness(sum_mix.permute(1, 0).numpy())
        lufs_delta_db = target_lufs_db - sum_lufs_db
        sum_mix = sum_mix * 10 ** (lufs_delta_db / 20)

        torchaudio.save(sum_filepath, sum_mix.view(1, -1), 44100)
        print("Sum mix saved.")

        # save the reference mix
        ref_filepath = os.path.join(example_dir, "ref-full.wav")
        torchaudio.save(ref_filepath, ref_audio.squeeze(), 44100)
        print("Reference mix saved.")

        for song_section in ["verse", "chorus"]:
            print("Mixing", song_section)
            if song_section == "verse":
                track_start_idx = example["track_verse_start_idx"]
                ref_start_idx = example["ref_verse_start_idx"]
            else:
                track_start_idx = example["track_chorus_start_idx"]
                ref_start_idx = example["ref_chorus_start_idx"]

            if track_start_idx + 262144 > tracks.shape[-1]:
                print("Tracks too short for this section.")
            if ref_start_idx + 262144 > ref_audio.shape[-1]:
                print("Reference too short for this section.")

            # crop the tracks to create a mix twice the size of the reference section
            mix_tracks = tracks
            # [..., track_start_idx : track_start_idx + (262144 * 2)]
            mix_tracks = tracks[..., track_start_idx : track_start_idx + (262144 * 2)]
            track_start_idx = 0
            print("mix_tracks", mix_tracks.shape)

            # save the reference mix section for analysis
            ref_analysis = ref_audio[..., ref_start_idx : ref_start_idx + 262144]

            # create mixes varying the loudness of the reference
            for ref_loudness_target in [-24, -16, -14.0, -12, -6]:
                print("Ref loudness", ref_loudness_target)
                ref_filepath = os.path.join(
                    example_dir,
                    f"ref-analysis-{song_section}-lufs-{ref_loudness_target:0.0f}.wav",
                )

                # loudness normalize the reference mix section to -14 LUFS
                ref_lufs_db = meter.integrated_loudness(
                    ref_analysis.squeeze().permute(1, 0).numpy()
                )
                lufs_delta_db = ref_loudness_target - ref_lufs_db
                ref_analysis = ref_analysis * 10 ** (lufs_delta_db / 20)

                torchaudio.save(ref_filepath, ref_analysis.squeeze(), 44100)

                for method_name, method in methods.items():
                    print(method_name)
                    # tracks (torch.Tensor): Set of input tracks with shape (bs, num_tracks, seq_len)
                    # ref_audio (torch.Tensor): Reference mix with shape (bs, 2, seq_len)

                    if method_name == "sum":
                        if ref_loudness_target != -16:
                            continue

                    if method_name == "sum" and song_section == "chorus":
                        continue

                    model, mix_console = method["model"]
                    func = method["func"]

                    print(tracks.shape, ref_audio.shape)

                    with torch.no_grad():
                        result = func(
                            mix_tracks.clone(),
                            ref_analysis.clone(),
                            model,
                            mix_console,
                            track_start_idx=track_start_idx,
                            ref_start_idx=ref_start_idx,
                        )

                        (
                            pred_mix,
                            pred_track_param_dict,
                            pred_fx_bus_param_dict,
                            pred_master_bus_param_dict,
                            pred_mixed_tracks,
                        ) = result

                    bs, chs, seq_len = pred_mix.shape

                    # loudness normalize the output mix
                    mix_lufs_db = meter.integrated_loudness(
                        pred_mix.squeeze(0).permute(1, 0).numpy()
                    )
                    print(mix_lufs_db)
                    lufs_delta_db = target_lufs_db - mix_lufs_db
                    pred_mix = pred_mix * 10 ** (lufs_delta_db / 20)

                    # save resulting audio and parameters
                    mix_filepath = os.path.join(
                        example_dir,
                        f"{example_name}-{method_name}-ref={song_section}-lufs-{ref_loudness_target:0.0f}.wav",
                    )
                    torchaudio.save(mix_filepath, pred_mix.view(chs, -1), 44100)

                    # also save only the analysis section
                    mix_analysis = pred_mix[
                        ..., track_start_idx : track_start_idx + (2 * 262144)
                    ]

                    # loudness normalize the output mix
                    mix_lufs_db = meter.integrated_loudness(
                        mix_analysis.squeeze(0).permute(1, 0).numpy()
                    )
                    print(mix_lufs_db)
                    mix_analysis = mix_analysis * 10 ** (lufs_delta_db / 20)

                    mix_filepath = os.path.join(
                        example_dir,
                        f"{example_name}-{method_name}-analysis-{song_section}-lufs-{ref_loudness_target:0.0f}.wav",
                    )
                    torchaudio.save(mix_filepath, mix_analysis.view(chs, -1), 44100)
                    
                    if method_name == "diffmst-16" and use_text_optimize:
                        track_idx = 0
                        text = "Make it sound brighter"
                        text_tokens = clap_model.get_text_tokens([text], device=device)
                        
                        # pred_mixed_tracks: batchsize, 2, num_tracks, seq_len
                        # projection layer: batchsize, 2*num_tracks, seq_len
                        # in inference, batchsize = 1
                        

                        # pred_mixed_tracks: (bs, 2, num_tracks, seq_len)
                        bs, chs, num_tracks, seq_len = pred_mixed_tracks.shape

                        track_embeddings = []
                        for b in range(bs):
                            audio_batch = []
                            for c in range(chs):
                                for t in range(num_tracks):
                                    # build a batch of mono waveforms for this track
                                    waveform = pred_mixed_tracks[b, c, t, :]
                                    audio_batch.append(waveform)
                            audio_batch = torch.stack(audio_batch, dim=0).to(device)  # (2*num_tracks, seq_len)
                            # get the embeddings for this batch
                            batch_track_embeddings = clap_model.get_audio_embedding(audio_batch, use_tensor=True) # (2*num_tracks, D)
                        
                            track_embeddings.append(batch_track_embeddings)
                        # stack into (bs, num_tracks, D)
                        track_embeddings = torch.stack(track_embeddings, dim=0) # (bs, 2*num_tracks, D)                        
                        
                        if optimize_option == "slerp":
                            alpha = 0.2
                            for i in range(2):
                                track_embedding = track_embeddings[0, i * num_tracks + track_idx, :]  # (D,)
                                text_token = text_tokens[0, :]  # (token_len,)

                                # normalize the embeddings
                                track_embedding_norm = track_embedding / track_embedding.norm(dim=0, keepdim=True)
                                text_token_norm = text_token / text_token.norm(dim=0, keepdim=True)

                                # slerp interpolation
                                omega = torch.acos(torch.clamp(torch.dot(track_embedding_norm, text_token_norm), -1.0, 1.0))
                                so = torch.sin(omega)
                                if so == 0:
                                    slerp_embedding = track_embedding
                                else:
                                    slerp_embedding = (torch.sin((1.0 - alpha) * omega) / so) * track_embedding + (torch.sin(alpha * omega) / so) * text_embedding

                                # replace the embedding
                                track_embeddings[0, i * num_tracks + track_idx, :] = slerp_embedding
                                
                            
                        elif optimize_option == "ADAM":
                            for i in range(2):
                                # Define the optimizer for the track embeddings
                                optimizer = torch.optim.Adam(
                                    [track_embeddings[0, i * num_tracks + track_idx, :]], lr=1e-3
                                )

                                # Optimization loop
                                for step in range(100):  # Number of optimization steps
                                    optimizer.zero_grad()

                                    # Compute the loss (example: cosine similarity loss)
                                    track_embedding = track_embeddings[0, i * num_tracks + track_idx, :]
                                    text_token = text_tokens[0, :]

                                    # Normalize the embeddings
                                    track_embedding_norm = track_embedding / track_embedding.norm(dim=0, keepdim=True)
                                    text_token_norm = text_token / text_token.norm(dim=0, keepdim=True)

                                    # Cosine similarity loss
                                    loss = -torch.dot(track_embedding_norm, text_token_norm)

                                    # Backpropagation
                                    loss.backward()
                                    optimizer.step()

                                print(f"Optimization completed for track {track_idx} with final loss: {loss.item()}")
                    else: 
                        pass
                        

        print()
