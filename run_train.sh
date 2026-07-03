CUDA_VISIBLE_DEVICES=0 python main.py fit \
  -c configs/config.yaml \
  -c configs/optimizer.yaml \
  -c configs/data/musdb18-v2.yaml \
  -c configs/models/naive+text.yaml \
  # --ckpt_path /work/ajchen2005/DiffMST_Retrain/1bm1am1y/checkpoints/epoch-040-step-1189.ckpt


# WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python main.py fit \
#   -c configs/config.yaml \
#   -c configs/optimizer.yaml \
#   -c configs/data/musdb18-v2.yaml \
#   -c configs/models/naive+text.yaml \
#   --trainer.fast_dev_run true \
#   # --trainer.logger false
