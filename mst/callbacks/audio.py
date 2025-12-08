import torch
import wandb
import numpy as np
import pyloudnorm as pyln
import pytorch_lightning as pl


from mst.callbacks.plotting import plot_spectrograms


class LogAudioCallback(pl.callbacks.Callback):
    def __init__(
        self,
        use_separate_tracks: bool = False,
        num_batches: int = 8,
        peak_normalize: bool = True,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.num_batches = num_batches
        self.peak_normalize = peak_normalize
        self.sample_rate = sample_rate
        self.use_separate_tracks = use_separate_tracks

        self.meter = pyln.Meter(sample_rate)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        """Called when the validation batch ends."""
        if outputs is not None:
            num_examples = outputs["ref_mix_a"].shape[0]
            if batch_idx < self.num_batches:
                for sample_idx in range(num_examples):
                    self.log_audio(
                        outputs,
                        self.use_separate_tracks,
                        batch_idx,
                        sample_idx,
                        pl_module.mix_console.sample_rate,
                        trainer.global_step,
                        trainer.logger,
                        f"Epoch {trainer.current_epoch}",
                    )

    def log_audio(
        self,
        outputs,
        use_separate_tracks: bool,
        batch_idx: int,
        sample_idx: int,
        sample_rate: int,
        global_step: int,
        logger,
        caption: str,
        n_fft: int = 4096,
        hop_length: int = 1024,
    ):
        audio_files = []
        audio_keys = []
        total_samples = 0
        # put all audio in file
        for key, audio in outputs.items():
            if "dict" in key:  # skip parameters
                continue

            x = audio[sample_idx, ...].float()
            if use_separate_tracks and key == "ref_mix_a":
                # reshape from (num_tracks*2, seq_len) to (2, num_tracks, seq_len)
                num_tracks = x.shape[0] // 2
                x = x.view(2, num_tracks, x.shape[1])
                # sum across tracks to get stereo mix
                x = x.sum(dim=1)
            x = x.permute(1, 0)
            # normalize the audio to -16 dBFS
            lufs_db = self.meter.integrated_loudness(x.numpy())
            delta_lufs_db = torch.tensor(
                [-16.0 - lufs_db]
            ).float()
            gain_lin = 10.0 ** (delta_lufs_db.clamp(-120, 40.0) / 20.0)
            x = x * gain_lin

            audio_files.append(x)
            audio_keys.append(key)
            total_samples += x.shape[0]

        y = torch.zeros(total_samples + int(len(audio_keys) * sample_rate), 2)
        name = f"{batch_idx}_{sample_idx}"
        start = 0
        for x, key in zip(audio_files, audio_keys):
            end = start + x.shape[0]
            y[start:end, :] = x
            start = end + int(sample_rate)
            name += key + "-"

        logger.experiment.log(
            {
                f"{name}": wandb.Audio(
                    y.numpy(),
                    caption=caption,
                    sample_rate=int(sample_rate),
                )
            }
        )

        # now try to log parameters
        pred_track_param_dict = outputs["pred_track_param_dict"]
        ref_track_param_dict = outputs["ref_track_param_dict"]

        pred_fx_bus_param_dict = outputs["pred_fx_bus_param_dict"]
        ref_fx_bus_param_dict = outputs["ref_fx_bus_param_dict"]

        pred_master_bus_param_dict = outputs["pred_master_bus_param_dict"]
        ref_master_bus_param_dict = outputs["ref_master_bus_param_dict"]

        effect_names = list(pred_track_param_dict.keys())

        column_names = None
        rows = []
        for effect_name in effect_names:
            param_names = list(pred_track_param_dict[effect_name].keys())
            for param_name in param_names:
                pred_param_val = pred_track_param_dict[effect_name][param_name]
                ref_param_val = ref_track_param_dict[effect_name][param_name]

                row = []
                row_name = f"{effect_name}.{param_name}"
                row.append(row_name)

                if column_names is None:
                    column_names = ["parameter"]
                    for i in range(pred_param_val.shape[1]):
                        column_names.append(f"{i}_pred")
                        column_names.append(f"{i}_ref")
                    # column_names.append("master_bus_pred")
                    # column_names.append("master_bus_ref")

                for i in range(pred_param_val.shape[1]):
                    row.append(pred_param_val[sample_idx, i].item())
                    row.append(ref_param_val[sample_idx, i].item())

                # row.append(pred_master_bus_param_dict[effect_name][batch_idx].item())

                rows.append(row)

        wandb_table = wandb.Table(data=rows, columns=column_names)
        logger.experiment.log(
            {f"batch={batch_idx}_sample={sample_idx}_parameters": wandb_table}
        )
