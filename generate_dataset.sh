#!/usr/bin/env bash
# Smoke test: 1 song, 1 augmentation
# python text_pair_gen/generate_dataset.py \
#   --output_dir /work/ajchen2005/musdb_smoke \
#   --track_root_dirs /work/ajchen2005/musdb18hq \
#   --max_songs 150 \
#   --augmentations 50

# python text_pair_gen/generate_dataset.py \
#   --output_dir /work/ajchen2005/musdb18_processed_v2 \
#   --config configs/data/musdb18-2.yaml \
#   --track_root_dirs /work/ajchen2005/musdb18hq \
#   --gemini_model gemini-flash-latest
#   --augmentations 10

python text_pair_gen/generate_dataset.py \
  --output_dir /work/ajchen2005/musdb18_processed_v2 \
  --config configs/data/musdb18-2.yaml \
  --track_root_dirs /work/ajchen2005/musdb18hq \
  --augmentations 10 \
  --gemini_model gemini-flash-latest \
  --start_song_idx 0 \
  --end_song_idx 29