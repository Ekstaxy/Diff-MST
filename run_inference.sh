#!/bin/bash

python test/inference.py \
    --config "configs/models/naive.yaml" \
    --checkpoint "/work/ajchen2005/DiffMST_Retrain/e4yhiogi/checkpoints/epoch=65-step=9900.ckpt" \
    --clap_checkpoint "../music_audioset_epoch_15_esc_90.14.pt" \
    --tracks_path "/work/ajchen2005/test_songs/" \
    --output_dir "/work/ajchen2005/inference_output/" \
    --tracks_start_idx 264600 \
    --ref_start_idx 264600 \
    --length 264600 \
    --num_tracks 11 \
    --target_lufs -14.0 \
    --ito_iterations 0 \
    --ito_lr 0.0002 \
    --prompt_str "bright vocal" \
    --neg_str "muffled vocal" \
    --target_track_idx 0
