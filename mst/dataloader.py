import os
import glob
import json
import torch
import yaml
import random
import itertools
import torchaudio
import numpy as np
import pyloudnorm as pyln
import pytorch_lightning as pl
from tqdm import tqdm
from typing import List

from torch.utils.data import random_split

class MixDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, length: int = 524288):
        super().__init__()
        self.root_dir = root_dir
        self.length = length

         
        self.mix_filepaths = glob.glob(
            os.path.join(root_dir, "**", "*.wav"), recursive=True)

        #self.mix_filepaths = glob.glob(
            #os.path.join(root_dir, "**", "*.mp3"), recursive=True)
        print(f"Located {len(self.mix_filepaths)} mixes.")

        self.meter = pyln.Meter(44100)

    def __len__(self):
        return len(self.mix_filepaths)

    def __getitem__(self, _):
        valid = False
        while not valid:
            # get random file
            idx = np.random.randint(0, len(self.mix_filepaths))
            # idx = 42  # always use the same mix for debug
            mix_filepath = self.mix_filepaths[idx]
            num_frames = torchaudio.info(mix_filepath).num_frames

            # find random non-silent region of the mix
            offset = np.random.randint(0, num_frames - self.length - 1)

            offset = 0  # always use the same offset


            mix, _ = torchaudio.load(
                mix_filepath,
                frame_offset=offset,
                num_frames=self.length,
            )

            if mix.shape[0] == 1:
                mix = mix.repeat(2, 1)
            elif mix.shape[0] > 2:
                mix = mix[:2, :]

            if mix.shape[-1] != self.length:
                continue

            mix_lufs_db = self.meter.integrated_loudness(mix.permute(1, 0).numpy())

            if mix_lufs_db > -48.0:
                valid = True

            # random gain of the target mixes
            target_lufs_db = np.random.randint(-48, 0)
            target_lufs_db = -14.0  # always use same target
            delta_lufs_db = torch.tensor([target_lufs_db - mix_lufs_db]).float()
            mix = 10.0 ** (delta_lufs_db / 20.0) * mix

        return mix


class MixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        length: int = 524288,
        num_workers: int = 4,
        batch_size: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()
        torchaudio.set_audio_backend("soundfile")

    def setup(self, stage=None):
        # create dataset
        dataset = MixDataset(self.hparams.root_dir, self.hparams.length)
        # create random splits
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [0.8, 0.1, 0.1]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=1,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=1,
        )


class MultitrackDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        track_root_dirs: List[str],
        metadata_files: List[str],
        mix_root_dirs: List[str] = [],
        instrument_id_json: str = "./data/instrument_name2id.json",
        sample_rate: int = 44100,
        length: int = 524288,
        min_tracks: int = 4,
        max_tracks: int = 20,
        subset: str = "train",
        buffer_size_gb: float = 0.01,
        target_track_lufs_db: float = -32.0,
        target_mix_lufs_db: float = -16.0,
        randomize_ref_mix_gain: bool = False,
        num_examples_per_epoch: int = 2500,
        num_passes: int = 1,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.length = length
        self.min_tracks = min_tracks
        self.max_tracks = max_tracks
        self.subset = subset
        self.buffer_size_gb = buffer_size_gb
        self.target_track_lufs_db = target_track_lufs_db
        self.target_mix_lufs_db = target_mix_lufs_db
        self.randomize_ref_mix_gain = randomize_ref_mix_gain
        self.meter = pyln.Meter(sample_rate)
        self.length = self.length
        self.num_passes = num_passes
        self.num_examples_per_epoch = num_examples_per_epoch

        with open(instrument_id_json, "r") as f:
            self.instrument_ids = json.load(f)

        self.song_dirs = {}
        self.dirs = []

        # load metadata for tracks
        for directory, split in zip(track_root_dirs, metadata_files):
            with open(split, "r") as f:
                data = yaml.safe_load(f)
                for songs, track_info in data[self.subset].items():
                    full_song_dir = os.path.join(directory, songs)
                    self.song_dirs[full_song_dir] = track_info
                    self.dirs.append(full_song_dir)

        print(f"Located {len(self.dirs)} track directories.")

        # load metadata for mixes
        self.mix_dirs = {}
        self.mixes = []

        for mix_dir in mix_root_dirs:
            # find all mixes in directory recursively

            mix_files = glob.glob(os.path.join(mix_dir, "**", "*.wav"), recursive=True)


            self.mixes.extend(mix_files)

        print(f"Located {len(self.mixes)} mixes.")

        self.num_examples = (
            self.num_examples_per_epoch + 1
        )  # this will trigger a reload of the buffer

    def __len__(self):
        return self.num_examples_per_epoch

    def reload_mix_buffer(self):
        self.mix_examples = []  # clear buffer
        nbytes_loaded = 0  # counter for data in RAM

        random.shuffle(self.mix_dirs)  # shuffle dataset

        pbar = tqdm(itertools.cycle(self.mixes))

        for filepath in pbar:
            num_frames = torchaudio.info(filepath, backend="soundfile").num_frames
            offset = np.random.randint(0.25 * num_frames, num_frames - self.length - 1)

            # ensure the song is long enough if we start from 25% in
            if (0.75 * num_frames) < self.length:
                continue

            mix, _ = torchaudio.load(
                filepath,
                frame_offset=offset,
                num_frames=self.length,
                backend="soundfile",
            )

            if mix.shape[0] == 1:
                continue
            if mix.shape[-1] != self.length:
                continue
            if mix.size()[0] > 2:
                continue

            mix_lufs_db = self.meter.integrated_loudness(mix.permute(1, 0).numpy())



            if mix_lufs_db < -48.0 or mix_lufs_db == float("-inf"):
                continue

            delta_lufs_db = torch.tensor(
                [self.target_mix_lufs_db - mix_lufs_db]
            ).float()

            gain_lin = 10.0 ** (delta_lufs_db.clamp(-120, 40.0) / 20.0)
            mix = gain_lin * mix

            self.mix_examples.append(mix)

            nbytes_loaded += mix.element_size() * mix.nelement()
            pbar.set_description(
                f"Loaded {nbytes_loaded/1e9:0.3f}/{self.buffer_size_gb} gb ({(nbytes_loaded/1e9/self.buffer_size_gb)*100:0.3f}%)"
            )

            # check if buffer is full
            if nbytes_loaded > self.buffer_size_gb * 1e9:
                break

    def reload_track_buffer(self):
        self.track_examples = []  # clear buffer
        nbytes_loaded = 0  # counter for data in RAM

        random.shuffle(self.dirs)  # shuffle dataset

        # load files into RAM
        pbar = tqdm(itertools.cycle(self.dirs))

        for dirname in pbar:
            track_filepaths = glob.glob(os.path.join(dirname, "*.wav"))

            song_name = os.path.basename(dirname)

            if len(track_filepaths) < self.min_tracks:
                continue
            random.shuffle(track_filepaths)

            num_frames = torchaudio.info(
                track_filepaths[0], backend="soundfile"
            ).num_frames

            middle_idx = int(num_frames / 2)

            # ensure the song is long enough if we start from 25% in
            if (0.75 * num_frames) < self.length:
                continue

            # load tracks
            tracks = []
            track_idx = 0
            track_metadata = []
            stereo_info = []
            track_padding = []
            # find a starting offset 25% into the song or more
            offset = np.random.randint(0.25 * num_frames, num_frames - self.length - 1)

            for track_filepath in track_filepaths:

                # ------------------------------------------------
                # Skip mixture.wav or files not in metadata
                filename = os.path.basename(track_filepath)
                if filename == "mixture.wav":
                    continue
                if filename not in self.song_dirs[dirname]:
                    continue
                # ------------------------------------------------

                stereo = False
                track, _ = torchaudio.load(
                    track_filepath,
                    frame_offset=offset,
                    num_frames=self.length,
                    backend="soundfile",
                )

                if track.shape[-1] != self.length:
                    continue
                if track.size()[0] > 2:
                    continue

                track_lufs_db = self.meter.integrated_loudness(
                    track.permute(1, 0).numpy()
                )

                if track_lufs_db < -48.0 or track_lufs_db == float("-inf"):
                    continue

                delta_lufs_db = torch.tensor(
                    [self.target_track_lufs_db - track_lufs_db]
                ).float()

                gain_lin = 10.0 ** (delta_lufs_db.clamp(-120, 40.0) / 20.0)
                track = gain_lin * track

                instrument = self.song_dirs[dirname][os.path.basename(track_filepath)]
                instrument = self.instrument_ids[instrument]

                if track.size()[0] == 2:
                    stereo = True

                for ch_idx in range(track.shape[0]):
                    if track_idx == self.max_tracks:
                        break
                    else:
                        tracks.append(track[ch_idx : ch_idx + 1, :])
                        track_metadata.append(instrument)
                        track_padding.append(False)
                        if stereo:
                            stereo_info.append(1)
                            stereo = False
                        else:
                            stereo_info.append(0)
                        track_idx += 1

                if track_idx >= self.max_tracks:
                    break

            if track_idx < self.min_tracks:
                continue

            # pad tracks to max_tracks
            while track_idx < self.max_tracks:
                tracks.append(torch.zeros_like(tracks[0]))
                track_metadata.append(0)
                track_padding.append(True)
                stereo_info.append(0)
                track_idx += 1

            # convert to tensor
            tracks = torch.cat(tracks)

            
            # if tracks[...,0:middle_idx].sum() == 0 or tracks[...,middle_idx:].sum() == 0:
            #     continue
            tracks = tracks.reshape(self.max_tracks, self.length)
            #create a sum mix of the tracks
            mix_check = tracks.sum(0)
            if torch.any(mix_check[...,0:middle_idx] == False) or torch.any(mix_check[...,middle_idx:] == False):
                continue

            track_metadata = torch.tensor(track_metadata)
            stereo_info = torch.tensor(stereo_info).reshape(track_metadata.shape)
            track_padding = torch.tensor(track_padding)

            # add to buffer
            self.track_examples.append(

                (tracks, stereo_info, track_metadata, track_padding, song_name)

            )

            nbytes_loaded += tracks.element_size() * tracks.nelement()
            pbar.set_description(
                f"Loaded {nbytes_loaded/1e9:0.3f}/{self.buffer_size_gb} gb ({(nbytes_loaded/1e9/self.buffer_size_gb)*100:0.3f}%)"
            )

            # check if buffer is full
            if nbytes_loaded > self.buffer_size_gb * 1e9:
                break

    def __getitem__(self, idx):

        # ----------- reload buffers if needed ------------
        if self.num_examples > self.num_examples_per_epoch:
            self.reload_track_buffer()
            self.reload_mix_buffer()
            self.num_examples = 0  # reset counter

        # ------------ get example from track buffer ------------
        track_example_idx = np.random.randint(0, len(self.track_examples))
        track_example = self.track_examples[track_example_idx]

        tracks = track_example[0]

        
        stereo_info = track_example[1]
        track_metadata = track_example[2]
        track_padding = track_example[3]
        song_name = track_example[4]


        # ------------ get example from mix buffer ------------
        # optional
        if len(self.mix_examples) > 0:
            mix_example_idx = np.random.randint(0, len(self.mix_examples))
            mix = self.mix_examples[mix_example_idx]

            if self.randomize_ref_mix_gain:
                gain_db = np.random.uniform(-16.0, 12.0)
                gain_lin = 10.0 ** (gain_db / 20.0)
                mix = gain_lin * mix
        else:
            mix = torch.empty(1)


        return tracks, stereo_info, track_metadata, track_padding, mix, song_name

class MultitrackDataModule(pl.LightningDataModule):
    def __init__(
        self,
        track_root_dirs: List[str],
        metadata_files: List[str],
        length: int,
        mix_root_dirs: List[str] = [],
        min_tracks: int = 4,
        max_tracks: int = 20,
        num_workers: int = 4,
        batch_size: int = 16,
        num_train_examples: int = 20000,
        num_val_examples: int = 1000,
        num_train_passes: int = 1,
        num_val_passes: int = 1,
        train_buffer_size_gb: float = 0.01,
        val_buffer_size_gb: float = 0.1,
        target_track_lufs_db: float = -48.0,
        target_mix_lufs_db: float = -16.0,
        randomize_ref_mix_gain: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        self.train_dataset = MultitrackDataset(
            track_root_dirs=self.hparams.track_root_dirs,
            metadata_files=self.hparams.metadata_files,
            mix_root_dirs=self.hparams.mix_root_dirs,
            subset="train",
            min_tracks=self.hparams.min_tracks,
            max_tracks=self.hparams.max_tracks,
            length=self.hparams.length,
            num_passes=self.hparams.num_train_passes,
            buffer_size_gb=self.hparams.train_buffer_size_gb,
            target_track_lufs_db=self.hparams.target_track_lufs_db,
            target_mix_lufs_db=self.hparams.target_mix_lufs_db,
            randomize_ref_mix_gain=self.hparams.randomize_ref_mix_gain,
        )

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        self.val_dataset = MultitrackDataset(
            track_root_dirs=self.hparams.track_root_dirs,
            metadata_files=self.hparams.metadata_files,
            mix_root_dirs=self.hparams.mix_root_dirs,
            subset="val",
            min_tracks=self.hparams.min_tracks,
            max_tracks=self.hparams.max_tracks,
            length=self.hparams.length,
            num_passes=self.hparams.num_val_passes,
            buffer_size_gb=self.hparams.val_buffer_size_gb,
            target_track_lufs_db=self.hparams.target_track_lufs_db,
            target_mix_lufs_db=self.hparams.target_mix_lufs_db,
            randomize_ref_mix_gain=self.hparams.randomize_ref_mix_gain,
        )

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=1,
        )

    def test_dataloader(self):
        self.test_dataset = MultitrackDataset(
            track_root_dirs=self.hparams.track_root_dirs,
            metadata_files=self.hparams.split,
            mix_root_dirs=self.hparams.mix_root_dirs,
            subset="test",
            min_tracks=self.hparams.min_tracks,
            max_tracks=self.max_tracks,
            length=self.hparams.length,
            num_passes=self.hparams.num_val_passes,
            buffer_size_gb=self.hparams.test_buffer_size_gb,
            target_track_lufs_db=self.hparams.target_track_lufs_db,
            target_mix_lufs_db=self.hparams.target_mix_lufs_db,
            randomize_ref_mix_gain=self.hparams.randomize_ref_mix_gain,
        )

        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=1,
        )
    
class PairedMixDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, metadata_file: str, split: str = "train", length: int = 524288, subset_ratio: float = 1.0, audio_drop_prob: float = 0.1, text_drop_prob: float = 0.1):
        super().__init__()
        self.length = length
        self.data_dir = data_dir
        self.subset_ratio = subset_ratio  # 0.5 means 50%
        self.audio_drop_prob = audio_drop_prob
        self.text_drop_prob = text_drop_prob
        
        # 1. 讀取 YAML 決定哪些歌屬於這個 split (train 或 val)
        with open(metadata_file, 'r') as f:
            meta = yaml.safe_load(f)
        allowed_songs = meta.get(split, [])
        
        # 2. 掃描所有符合條件的 augmentations
        self.all_samples = []
        for song in allowed_songs:
            song_dir = os.path.join(data_dir, song)
            if not os.path.isdir(song_dir):
                continue
            
            param_files = glob.glob(os.path.join(song_dir, "aug_*_params.pt"))
            for pf in param_files:
                base_name = os.path.basename(pf).replace("_params.pt", "") 
                
                # [修改 1]：檢查 Vocal 和 Instrumental 的 JSON 是否都存在
                vocal_json = os.path.join(song_dir, f"{base_name}_compare_wet_dry_vocal.json")
                inst_json = os.path.join(song_dir, f"{base_name}_compare_wet_dry_instrumental.json")
                
                if not (os.path.exists(vocal_json) and os.path.exists(inst_json)):
                    continue
                    
                self.all_samples.append({
                    "song_name": song,
                    "song_dir": song_dir,
                    "base_name": base_name,
                    "param_path": pf
                })
        
        self.shuffle_and_subset()

    def shuffle_and_subset(self):
        """Shuffle all samples and pick a subset for this epoch."""
        random.shuffle(self.all_samples)
        # Use config-defined ratio
        subset_size = int(len(self.all_samples) * self.subset_ratio)
        
        # Ensure at least one sample if possible
        if subset_size == 0 and len(self.all_samples) > 0:
            subset_size = len(self.all_samples)
            
        self.samples = self.all_samples[:subset_size]
        print(f"Epoch subset: Selected {len(self.samples)}/{len(self.all_samples)} samples (Ratio: {self.subset_ratio}).")

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        song_name = sample["song_name"]
        base_name = sample["base_name"]
        song_dir = sample["song_dir"]
        
        # 1. Read Ground Truth 參數 (.pt)
        params = torch.load(sample["param_path"], weights_only=True)
        
        # 2. Read Dry Tracks (Track Input) 從 V2 的 dry 子資料夾讀取
        dry_vocal_path = os.path.join(song_dir, "dry", f"{base_name}_vocal.wav")
        dry_inst_path = os.path.join(song_dir, "dry", f"{base_name}_instrumental.wav")
        
        dry_vocal, _ = torchaudio.load(dry_vocal_path)
        dry_inst, _ = torchaudio.load(dry_inst_path)

        # Make sure it is mono
        if dry_vocal.shape[0] > 1: dry_vocal = dry_vocal.mean(dim=0, keepdim=True)
        if dry_inst.shape[0] > 1: dry_inst = dry_inst.mean(dim=0, keepdim=True)

        tracks = torch.cat([dry_inst, dry_vocal], dim=0) # [Other, Vocal]
        
        # 3. Read Source Separation Estimate (Refer Input) 從 V2 的 src_sep 讀取
        vocals_est_path = os.path.join(song_dir, "src_sep", f"{base_name}_vocal.wav")
        other_est_path = os.path.join(song_dir, "src_sep", f"{base_name}_instrumental.wav")
        
        vocals_est, _ = torchaudio.load(vocals_est_path)
        other_est, _ = torchaudio.load(other_est_path)
        
        # if vocals_est.shape[0] > 1: vocals_est = vocals_est.mean(dim=0, keepdim=True)
        # if other_est.shape[0] > 1: other_est = other_est.mean(dim=0, keepdim=True)
        
        est_tracks = torch.cat([other_est, vocals_est], dim=0)

        # 4. Read Ground Truth Mix
        mix_path = os.path.join(song_dir, f"{base_name}_mix.wav")
        true_mix, _ = torchaudio.load(mix_path)
        
        # 5. Read Text Prompt
        vocal_json_path = os.path.join(song_dir, f"{base_name}_compare_wet_dry_vocal.json")
        inst_json_path = os.path.join(song_dir, f"{base_name}_compare_wet_dry_instrumental.json")

        def load_and_sample_json(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            keys = ["gain", "pan", "compressor", "eq"]
            
            # [關鍵修改 1]：隨機決定這次要取用幾個效果器的描述 (1 到 1 個)
            # num_traits_to_keep = random.randint(1, 2)
            # 隨機抽取這些 keys
            selected_keys = random.sample(keys, 1)
            
            parts = []
            for k in selected_keys:
                val = data.get(k, "").strip()
                if val and val != "(no description generated)":
                    # [關鍵修改 2]：移除句尾的句號，並把首字母轉小寫，讓句子更通順
                    val = val.rstrip(".")
                    if val:
                        val = val[0].lower() + val[1:]
                    parts.append(val)
            
            if not parts:
                return ""
                
            # [關鍵修改 3]：用自然的連接詞合併 (A, B, and C)
            if len(parts) == 1:
                return parts[0]
            elif len(parts) == 2:
                return f"{parts[0]} and {parts[1]}"
            else:
                return ", ".join(parts[:-1]) + f", and {parts[-1]}"
        
        vocal_text = load_and_sample_json(vocal_json_path)
        inst_text = load_and_sample_json(inst_json_path)

        # 為了避免抽完剛好全是空字串導致出錯，給個預設值
        vocal_text = vocal_text if vocal_text else "exactly as it is"
        inst_text = inst_text if inst_text else "exactly as it is"

        # (接下來的 template_choice 邏輯維持你原本的寫法即可)
        template_choice = random.random()
        if template_choice < 0.25:
            text = f"Make the vocal {vocal_text}, and keep the instrumental {inst_text}."
        elif template_choice < 0.50:
            text = f"The vocal sounds {vocal_text}, while the instrumental is {inst_text}."
        elif template_choice < 0.75:
            text = f"Push the instrumental to be {inst_text}, and make sure the vocal is {vocal_text}."
        elif template_choice < 0.90:
            if random.random() > 0.5:
                text = f"Just make the vocal {vocal_text}."
            else:
                text = f"I want the instrumental to be {inst_text}."
        else:
            text = f"Vocal is {vocal_text}. Instrumental is {inst_text}."

        # print(f"Generated Text Prompt: {text}")
        
        # Helper to crop/pad
        def process(t, length, off):
            if t.shape[-1] > length:
                return t[..., off:off+length]
            elif t.shape[-1] < length:
                return torch.nn.functional.pad(t, (0, length - t.shape[-1]))
            return t

        # Use tracks length to determine offset
        current_len = tracks.shape[-1]
        offset = 0
        if current_len > self.length:
            offset = np.random.randint(0, current_len - self.length)
        
        tracks = process(tracks, self.length, offset)[..., self.length//2:self.length] # 只取後半段
        est_tracks = process(est_tracks, self.length, offset)[..., :self.length//2]
        true_mix = process(true_mix, self.length, offset)[..., self.length//2:self.length]
        
        # --- Modality Dropout (Classifier-Free Guidance) ---
        if random.random() < self.text_drop_prob:
            text = ""  # Drop text condition
            
        if random.random() < self.audio_drop_prob:
            est_tracks = torch.zeros_like(est_tracks)  # Drop audio condition
        
        # 5. 給 system.py 的佔位符 (Dummy data)
        stereo_info = torch.tensor([0, 0])
        track_padding = torch.tensor([False, False])
        
        # Return 8 items matching updated system.py unpacking:
        # tracks, est_tracks, true_mix, stereo_info, track_padding, song_name, ref_params_dict, text
        return tracks, est_tracks, true_mix, stereo_info, track_padding, song_name, params, text


class PairedMixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        length: int = 524288,
        batch_size: int = 32,
        num_workers: int = 4,
        train_subset_ratio: float = 1.0,
        val_subset_ratio: float = 1.0,
        audio_drop_prob: float = 0.1,
        text_drop_prob: float = 0.1
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_dataset = PairedMixDataset(
            data_dir=self.hparams.data_dir, 
            metadata_file=self.hparams.metadata_file, 
            split="train", 
            length=self.hparams.length,
            subset_ratio=self.hparams.train_subset_ratio,
            audio_drop_prob=self.hparams.audio_drop_prob,
            text_drop_prob=self.hparams.text_drop_prob
        )
        self.val_dataset = PairedMixDataset(
            data_dir=self.hparams.data_dir, 
            metadata_file=self.hparams.metadata_file, 
            split="val", 
            length=self.hparams.length,
            subset_ratio=self.hparams.val_subset_ratio,
            audio_drop_prob=self.hparams.audio_drop_prob,
            text_drop_prob=self.hparams.text_drop_prob
        )

    def train_dataloader(self):
        # Reshuffle dataset before creating dataloader for the new epoch
        if hasattr(self.train_dataset, 'shuffle_and_subset'):
             self.train_dataset.shuffle_and_subset()
             
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)


# if __name__ == "__main__":

#     dataset = multitrack_dataset()
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

#     for i, (tracks, stereo_info, track_metadata) in enumerate(dataloader):
#         print(tracks.shape)
#         print(stereo_info.shape)
#         print(track_metadata.shape)
#         break
