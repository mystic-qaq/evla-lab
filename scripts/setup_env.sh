#!/usr/bin/env bash
# One-shot environment setup for a GPU workstation (tested on 8x RTX 3090, CUDA 12.8).
#
# Why a script: the dependency graph here is genuinely fiddly, and every choice below
# comes from a failure mode that cost time:
#   * transformers must be <5 and >=4.57 for SmolVLA's SmolVLM2 backbone
#   * torchcodec needs system FFmpeg 6/7, which most lab images lack -> we use PyAV
#   * the VLM weights come from the Hub, and direct HF access is blocked on many
#     Chinese networks, hence HF_ENDPOINT
set -euo pipefail

PY=${PY:-3.10}
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VENV=${VENV:-$ROOT/.venv}
INDEX=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[setup] venv (python $PY) -> $VENV"
uv venv --python "$PY" "$VENV"

PYBIN="$VENV/bin/python"
echo "[setup] installing torch (CUDA) + lerobot"
uv pip install --python "$PYBIN" --index-strategy unsafe-best-match \
  torch torchvision --index-url "$INDEX"
uv pip install --python "$PYBIN" --index-strategy unsafe-best-match \
  "lerobot[smolvla]==0.4.4" "gym-pusht==0.1.8" pygame imageio imageio-ffmpeg matplotlib --index-url "$INDEX"

echo "[setup] installing evla (editable)"
uv pip install --python "$PYBIN" --no-deps -e "$ROOT"

cat <<'EOF'

[setup] done. Add this to your shell profile so Hub downloads work:

    export HF_ENDPOINT=https://hf-mirror.com

Then:
    evla doctor
    evla quickstart --out outputs/quickstart      # CPU, ~1 min, no GPU needed
    evla finetune -c configs/finetune_smolvla_pusht.yaml
EOF
