#!/usr/bin/env bash
# Fast sanity check of every stage without a GPU: 1-2 minutes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
.venv/bin/python -m pytest tests -q
.venv/bin/evla quickstart --out outputs/quickstart --episodes 40 --steps 400 --eval-episodes 8
