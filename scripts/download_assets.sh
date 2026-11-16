#!/usr/bin/env bash
# Fetch the public checkpoint and dataset used by the experiments.
# Set HF_ENDPOINT=https://hf-mirror.com if huggingface.co is unreachable.
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-$ROOT/.venv/bin/python}

$PY - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("lerobot/smolvla_base", local_dir="checkpoints/smolvla_base")
snapshot_download("lerobot/pusht", repo_type="dataset", local_dir="data/pusht")
print("assets ready: checkpoints/smolvla_base (873 MB), data/pusht (7.5 MB)")
PY
