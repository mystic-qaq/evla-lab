"""Policy loading from config dictionaries."""

from __future__ import annotations

from typing import Any

from evla.types import Policy


def load_policy(cfg: dict[str, Any]) -> Policy:
    """Instantiate a policy.

    ``cfg`` shapes::

        {"type": "smolvla", "checkpoint": "...", "device": "cuda", "precision": "bf16"}
        {"type": "flow_chunk", "checkpoint": "outputs/toy_flow.npz", "device": "cpu"}
    """
    cfg = dict(cfg or {})
    kind = cfg.pop("type", None)
    if kind is None:
        kind = "smolvla" if "checkpoint" in cfg else "flow_chunk"

    if kind == "smolvla":
        from evla.models.smolvla import SmolVLAPolicy

        allowed = {"checkpoint", "device", "precision", "num_flow_steps", "image_key", "rename"}
        _reject_unknown(cfg, allowed, kind)
        return SmolVLAPolicy(**cfg)

    if kind in ("flow", "flow_chunk"):
        from evla.models.flow_chunk import FlowChunkPolicy

        path = cfg.pop("checkpoint", cfg.pop("path", None))
        if path is None:
            raise ValueError("flow_chunk policy needs a 'checkpoint' path")
        device = cfg.pop("device", "cpu")
        policy = FlowChunkPolicy.load(path, device=device)
        for key, value in cfg.items():
            if not hasattr(policy, key):
                raise ValueError(f"flow_chunk has no attribute {key!r}")
            setattr(policy, key, value)
        return policy

    raise ValueError(f"unknown policy type {kind!r}")


def _reject_unknown(cfg: dict[str, Any], allowed: set[str], kind: str) -> None:
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"unknown keys for {kind} policy: {sorted(unknown)} (allowed: {sorted(allowed)})")
