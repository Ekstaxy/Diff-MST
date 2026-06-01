#!/bin/bash

# Remember to change the dropout to 0 if using old checkpoints that were trained without dropout, since the model architecture has changed.

PYTHONPATH=. python test/inference.py \
    --config "configs/models/naive.yaml" \
    --checkpoint "/work/ajchen2005/DiffMST_Retrain/gxuqbygb/checkpoints/epoch=151-step=22770.ckpt" \
    --clap_checkpoint "../music_audioset_epoch_15_esc_90.14.pt" \
    --tracks_path "/work/ajchen2005/test_songs/" \
    --output_dir "/work/ajchen2005/inference_outputs_AB_test" \
    --ref_input_type "audio" \
    --ito_target_type "audio" \
    --tracks_start_idx 264600 \
    --ref_start_idx 264600 \
    --length 264600 \
    --num_tracks 12 \
    --target_lufs -14.0 \
    --ito_iterations 100 \
    --ito_lr 0.0002 \
    --prompt_str "bright vocal" \
    --neg_str "dark" \
    --target_track_idx 1
