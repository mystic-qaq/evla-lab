"""Capability-probing optimization pipeline.

The point of this module is to make each optimization an explicit, individually
attributable knob instead of a mystery "it got faster" change. A policy declares
what it supports by exposing methods (``set_precision``, ``set_num_flow_steps``)
or an ``nn.Module`` behind ``.policy``; :func:`apply_optimizations` composes the
requested knobs, records which ones actually took effect, and never silently
pretends an unsupported knob was applied.

Optimization axes that actually move control frequency for a VLA:

===============  ==========================================================
precision        fp32 -> bf16/fp16 matmuls. On Ampere+ this is the cheapest
                 large win, and it does not change the algorithm.
flow steps       Euler steps of the flow-matching action expert (10 -> 4 -> 2).
                 Linear in latency; the accuracy cost is the interesting part.
compile          ``torch.compile`` on the denoise loop. Big win on paper,
                 frequently blocked by dynamic shapes -- hence the guard.
chunking         Not here: handled by :class:`evla.control.ChunkExecutor`,
                 because it changes the *control* strategy, not the model.
===============  ==========================================================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from evla.types import Observation, Policy


@dataclass
class OptimizationSpec:
    precision: str = "fp32"
    num_flow_steps: int | None = None
    compile: bool = False
    compile_mode: str = "default"
    threads: int | None = None
    label: str = ""

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> OptimizationSpec:
        cfg = dict(cfg or {})
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"unknown optimization keys: {sorted(unknown)}")
        return cls(**cfg)

    def key(self) -> str:
        if self.label:
            return self.label
        parts = [self.precision]
        if self.num_flow_steps:
            parts.append(f"flow{self.num_flow_steps}")
        if self.compile:
            parts.append(f"compile-{self.compile_mode}")
        return "+".join(parts)

    def describe(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, False, "")}


class CompiledPolicy(Policy):
    """``torch.compile`` wrapper around a torch-module-backed policy.

    Compilation is attempted lazily on the first call so that failures surface as
    a recorded status instead of a crash during construction.
    """

    def __init__(self, inner: Policy, mode: str = "default") -> None:
        import torch

        self.inner = inner
        self.name = f"{inner.name}+compiled"
        self.action_dim = inner.action_dim
        self.chunk_size = inner.chunk_size
        self.status = "pending"
        module = getattr(inner, "policy", None)
        if not isinstance(module, torch.nn.Module):
            self.status = "unsupported: inner policy exposes no nn.Module"
            return
        try:
            module.forward = torch.compile(module.forward, mode=mode, fullgraph=False)  # type: ignore[method-assign]
            self.status = "compiled"
        except Exception as exc:  # pragma: no cover - environment dependent
            self.status = f"failed: {type(exc).__name__}: {exc}"

    def reset(self, task: str = "") -> None:
        self.inner.reset(task)

    def predict_chunk(self, obs: Observation):
        try:
            return self.inner.predict_chunk(obs)
        except Exception as exc:  # pragma: no cover - compilation boundaries
            self.status = f"runtime-error: {type(exc).__name__}: {exc}"
            raise

    def describe(self) -> dict:
        info = self.inner.describe()
        info["compile_status"] = self.status
        return info


@dataclass
class AppliedOptimizations:
    applied: dict[str, Any] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"applied": self.applied, "skipped": self.skipped}


def apply_optimizations(policy: Policy, spec: OptimizationSpec) -> tuple[Policy, AppliedOptimizations]:
    """Apply ``spec`` to ``policy`` in place, returning the (possibly wrapped) policy."""
    result = AppliedOptimizations()

    if spec.threads is not None:
        try:
            import torch

            torch.set_num_threads(int(spec.threads))
            result.applied["threads"] = int(spec.threads)
        except Exception as exc:
            result.skipped["threads"] = f"{type(exc).__name__}: {exc}"

    if spec.precision != "fp32":
        if hasattr(policy, "set_precision"):
            policy.set_precision(spec.precision)  # type: ignore[attr-defined]
            result.applied["precision"] = spec.precision
        else:
            result.skipped["precision"] = "policy has no set_precision()"

    if spec.num_flow_steps is not None:
        if hasattr(policy, "set_num_flow_steps"):
            policy.set_num_flow_steps(int(spec.num_flow_steps))  # type: ignore[attr-defined]
            result.applied["num_flow_steps"] = int(spec.num_flow_steps)
        else:
            result.skipped["num_flow_steps"] = "policy has no set_num_flow_steps()"

    if spec.compile:
        wrapped = CompiledPolicy(policy, mode=spec.compile_mode)
        policy = wrapped
        if wrapped.status == "compiled":
            result.applied["compile"] = spec.compile_mode
        else:
            result.skipped["compile"] = wrapped.status

    return policy, result
