#!/usr/bin/env bash
# The full evaluation matrix: fine-tuning quality x execution mode x precision x flow steps.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
.venv/bin/evla sweep -c configs/sweep_pusht.yaml --out results/pusht_sweep
