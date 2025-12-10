import joblib
import pathlib
import argparse
import torchaudio
from tqdm import tqdm
from contextlib import contextmanager

@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Resample audio files to a target sample rate.")
    parser.add_argument(
        "--input_dir",
        type=pathlib.Path,
        help="Directory containing input audio files.",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        help="Directory to save resampled audio files.",
    )
    parser.add_argument(
        "--target_sample_rate",
        type=int,
        default=48000,
        help="Target sample rate for resampling.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of parallel workers for processing."
    )
    return parser.parse_args()

def resample_file(input_file, output_file, target_sample_rate):
    try:
        waveform, original_sample_rate = torchaudio.load(input_file, backend="soundfile")
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate,
            new_freq=target_sample_rate,
        )
        resampled_waveform = resampler(waveform)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_file, resampled_waveform, target_sample_rate)
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def main():
    args = parse_args()

    # Find all .wav files in the input directory & its subdirectories
    input_files = list(args.input_dir.rglob("*.wav"))
    with tqdm_joblib(tqdm(total=len(input_files), desc="Resampling audio files")):
        joblib.Parallel(n_jobs=args.num_workers, verbose=0)(
            joblib.delayed(resample_file)(
                input_file,
                args.output_dir / input_file.relative_to(args.input_dir),
                args.target_sample_rate
            ) for input_file in input_files
        )

if __name__ == "__main__":
    main()