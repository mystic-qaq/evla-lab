#!/usr/bin/env bash
# Full experimental programme, in order. Total ~5 h on one RTX 3090.
#
#   1. fine-tune (20k steps)            ~4.3 h
#   2. learning curve (5 ckpts x 20 ep)  ~12 min
#   3. optimization matrix (8 configs)   ~25 min
#   4. deployment run in the simulator    ~2 min
#
# Every stage writes into results/ so the tables in the README can be regenerated
# without re-running the GPU work.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=.venv/bin/evla

echo "== 0. environment =="
$PY doctor

echo "== 1. fine-tune SmolVLA on Push-T =="
$PY finetune -c configs/finetune_smolvla_pusht.yaml

echo "== 2. success vs amount of fine-tuning =="
$PY sweep -c configs/sweep_learning_curve.yaml --out results/pusht_learning_curve

echo "== 3. optimization / execution matrix =="
$PY sweep -c configs/sweep_pusht.yaml --out results/pusht_sweep

echo "== 4. deployment runtime (simulator) =="
$PY deploy -c configs/deploy.yaml --episodes 3 --out results/deploy_sim.json

echo "== done =="
cat results/pusht_sweep/REPORT.md
