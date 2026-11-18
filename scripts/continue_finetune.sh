#!/usr/bin/env bash
# Stage-2 fine-tune: continue from the 20k checkpoint for another 10k steps at a
# reduced learning rate.
#
# Why: the 20k run's coverage plateaus below the 0.95 success threshold, which is a
# precision/behaviour problem rather than an optimisation problem. A warm restart at
# half the LR is the cheapest way to test whether the remaining gap is a training
# budget issue or a capacity/data issue. If coverage does not move, the honest
# conclusion is "more steps of the same data do not fix pose precision", and that is
# what the report says.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

.venv/bin/python -m evla.models.smolvla_train \
  --policy.path=outputs/smolvla_pusht_20k/checkpoints/020000/pretrained_model \
  --policy.repo_id=local/evla-smolvla-pusht-30k --policy.push_to_hub=false \
  --policy.empty_cameras=2 --policy.device=cuda \
  --policy.chunk_size=50 --policy.n_action_steps=50 \
  --policy.optimizer_lr=5e-5 \
  --dataset.repo_id=lerobot/pusht --dataset.root=data/pusht --dataset.video_backend=pyav \
  --rename_map='{"observation.image": "observation.images.camera1"}' \
  --batch_size=32 --steps=10000 --save_freq=2500 --log_freq=200 \
  --output_dir=outputs/smolvla_pusht_30k --job_name=smolvla_pusht_30k \
  --num_workers=6 --wandb.enable=false
