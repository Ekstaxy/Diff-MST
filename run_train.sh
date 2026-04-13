CUDA_VISIBLE_DEVICES=0 python main.py fit \
  -c configs/config.yaml \
  -c configs/optimizer.yaml \
  -c configs/data/musdb18-2.yaml \
  -c configs/models/naive.yaml