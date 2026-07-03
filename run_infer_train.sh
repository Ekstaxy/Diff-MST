#!/bin/bash
# Inference on musdb18_processed_v2 using the current naive+text.yaml (3-layer controller).
#
# Usage:
#   CHECKPOINT=/path/to/your.ckpt bash run_infer_train.sh
# Or edit CHECKPOINT below after the new 3-layer run saves a checkpoint.

set -euo pipefail

CONFIG="configs/models/naive+text.yaml"
OUTPUT_DIR="/work/ajchen2005/inference_train_samples_3layer"
CHECKPOINT="${CHECKPOINT:-/work/ajchen2005/DiffMST_Retrain/2ft82nm8/checkpoints/last.ckpt}"

PYTHONPATH=. python test/infer_train_sample.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --split train \
  --aug_idx 0 \
  --num_samples 3 \
  --ref_input_type text \
  --match_training_crop \
  --start_idx 0 \
  --output_dir "${OUTPUT_DIR}"

echo "Done. Listen under: ${OUTPUT_DIR}/train/<song>/aug_0/"
