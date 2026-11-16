"""Merge a PEFT/LoRA adapter produced by ``lerobot-train --peft.*`` into a checkpoint.

Why this exists: LeRobot saves LoRA fine-tunes as PEFT adapters
(``adapter_model.safetensors``, ~3 MB) rather than full weights, and its
``from_pretrained`` cannot load them back on its own. Merging gives a normal
``pretrained_model`` directory that the rest of this pipeline (evaluation,
optimization, serving) can use unchanged.

The dangerous part is *silence*: ``load_state_dict(..., strict=False)`` happily
accepts a key mismatch and you end up merging an all-zero adapter -- the merged model
is then just the base model, and every downstream number is quietly wrong. This script
therefore **asserts that every LoRA tensor was consumed** before merging, and can
verify that the merged model's outputs actually differ from the base model's.

    python scripts/merge_lora_adapter.py \
        --base checkpoints/smolvla_base \
        --adapter outputs/smolvla_pusht_lora8k/checkpoints/004000/pretrained_model \
        --out outputs/smolvla_pusht_lora8k/merged_004000 \
        --rank 16 --verify
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

import numpy as np


def _normalise_keys(state_dict: dict) -> dict:
    """PEFT stores unnamed adapters as ``lora_A.weight``; the wrapped model expects
    ``lora_A.default.weight``. Normalise both directions so either style loads."""
    out = {}
    for key, value in state_dict.items():
        new = key
        for marker in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
            if f"{marker}.weight" in key:
                new = key.replace(f"{marker}.weight", f"{marker}.default.weight")
        out[new] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="base pretrained_model directory")
    parser.add_argument("--adapter", required=True, help="PEFT adapter directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--method", default="LORA")
    parser.add_argument("--verify", action="store_true", help="check outputs changed after merging")
    args = parser.parse_args()

    import torch
    from lerobot.configs.default import PeftConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from safetensors.torch import load_file

    # Take the *architecture config* from the adapter (it knows the fine-tuned action
    # dimensionality), and only the weights from the base checkpoint. Using the base
    # config silently produces a 6-DoF head for a 2-DoF task.
    config = PreTrainedConfig.from_pretrained(args.adapter)
    policy = SmolVLAPolicy.from_pretrained(args.base, config=config)
    policy.config.pretrained_path = args.base
    wrapped = policy.wrap_with_peft(
        peft_cli_overrides=dataclasses.asdict(PeftConfig(method_type=args.method, r=args.rank))
    )

    adapter = _normalise_keys(load_file(f"{args.adapter}/adapter_model.safetensors"))
    expected = {k for k in wrapped.state_dict() if "lora_" in k}
    provided = {k for k in adapter if "lora_" in k}
    missing = expected - provided
    extra = provided - expected
    print(f"[merge] adapter tensors: {len(provided)}, expected: {len(expected)}")
    if missing or extra:
        print(f"[merge] ERROR missing={sorted(missing)[:3]} extra={sorted(extra)[:3]}")
        return 2

    result = wrapped.load_state_dict(adapter, strict=False)
    unresolved = [k for k in result.unexpected_keys if "lora_" in k]
    unmatched = [k for k in result.missing_keys if "lora_" in k]
    if unresolved or unmatched:
        print(f"[merge] ERROR lora keys not consumed: {unresolved[:3]} {unmatched[:3]}")
        return 2

    before = None
    batch = None
    if args.verify:
        from lerobot.policies.factory import make_pre_post_processors

        preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=args.base)
        batch = preprocessor(_dummy_batch())
        policy.eval()
        with torch.inference_mode():
            before = policy.predict_action_chunk(batch).float().cpu().numpy()

    merged = wrapped.merge_and_unload()
    merged.save_pretrained(args.out)
    # The normalisation statistics live in the processor files, not in the weights:
    # copy them so the output is a drop-in replacement for a `pretrained_model` dir.
    copied = _copy_processors(args.adapter, args.out)
    print(f"[merge] saved merged checkpoint -> {args.out} (copied {copied} processor files)")

    if before is not None:
        merged.eval()
        with torch.inference_mode():
            after = merged.predict_action_chunk(batch).float().cpu().numpy()
        delta = float(np.abs(after - before).max())
        print(f"[merge] max |delta action| before vs after merge: {delta:.4f}")
        if delta < 1e-6:
            print("[merge] WARNING: merged model behaves identically to the base -- adapter had no effect")
            return 3
    return 0


def _copy_processors(adapter_dir: str, out_dir: str) -> int:
    import shutil
    from pathlib import Path

    src, dst = Path(adapter_dir), Path(out_dir)
    n = 0
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        path = src / name
        if path.exists():
            shutil.copy2(path, dst / name)
            n += 1
    for path in src.glob("policy_pre*processor_step_*.safetensors"):
        shutil.copy2(path, dst / path.name)
        n += 1
    for path in src.glob("policy_post*processor_step_*.safetensors"):
        shutil.copy2(path, dst / path.name)
        n += 1
    return n


_DUMMY = {}


def _dummy_batch() -> dict:
    """Fixed-shape, deterministic batch so before/after comparison is meaningful."""
    if _DUMMY:
        return _DUMMY
    import torch

    _DUMMY.update(
        {
            "observation.images.camera1": torch.zeros(1, 3, 96, 96),
            "observation.state": torch.zeros(1, 2),
            "task": ["Push the T-shaped block onto the T-shaped target."],
        }
    )
    return _DUMMY


if __name__ == "__main__":
    sys.exit(main())
