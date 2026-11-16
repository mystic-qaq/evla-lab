.PHONY: help setup assets test quickstart finetune sweep serve deploy report lint

help:
	@echo "evla-lab"
	@echo "  make setup      create .venv and install torch + lerobot + evla"
	@echo "  make assets     download SmolVLA base checkpoint and Push-T dataset"
	@echo "  make test       run the test-suite (CPU)"
	@echo "  make quickstart CPU-only collect -> train -> eval -> report"
	@echo "  make finetune   fine-tune SmolVLA on Push-T (needs a GPU)"
	@echo "  make sweep      run the optimization/execution matrix"
	@echo "  make serve      serve the fine-tuned policy over TCP"
	@echo "  make deploy     run the fixed-rate runtime against the simulator"

setup:
	bash scripts/setup_env.sh

assets:
	bash scripts/download_assets.sh

test:
	.venv/bin/python -m pytest tests -q

lint:
	.venv/bin/ruff check src tests

quickstart:
	.venv/bin/evla quickstart --out outputs/quickstart

finetune:
	bash scripts/run_finetune.sh

sweep:
	bash scripts/run_sweep.sh

serve:
	.venv/bin/evla serve -c configs/deploy.yaml --port 5599

deploy:
	.venv/bin/evla deploy -c configs/deploy.yaml
