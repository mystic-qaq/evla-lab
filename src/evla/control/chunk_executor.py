"""Turn a chunk-predicting policy into a per-step controller.

This module is where most of the *real* deployment trade-off lives. A policy that
predicts :math:`H` future actions can be executed in three ways:

``receding``
    Re-run inference at every control step and keep only the first action.
    Maximum reactivity, maximum compute. This is what OpenVLA-style single-step
    VLAs do naively.

``open_loop``
    Execute the chunk without re-planning, then re-plan. Inference cost is
    amortised over ``H`` steps, so the effective control rate jumps by ~H times,
    but the robot is blind for the rest of the chunk.

``temporal_ens``
    Re-plan every step but average all overlapping predictions with an
    exponential recency weighting (ACT, Zhao et al. 2023). Smooth trajectories at
    the cost of always paying inference.

Measuring all three on the same policy is how you answer the question the
competition actually asks: *how fast can the loop run before success rate drops?*
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from evla.types import Observation, Policy

MODES = ("receding", "open_loop", "temporal_ens")


@dataclass
class ExecutorStats:
    n_steps: int = 0
    n_inference: int = 0
    per_step_inference: list[float] = field(default_factory=list)

    @property
    def reuse_ratio(self) -> float:
        if self.n_inference == 0:
            return 0.0
        return self.n_steps / self.n_inference

    def as_dict(self) -> dict[str, float]:
        return {
            "n_steps": self.n_steps,
            "n_inference": self.n_inference,
            "action_reuse": round(self.reuse_ratio, 3),
        }


class ChunkExecutor:
    """Drive a :class:`~evla.types.Policy` at every environment step.

    Args:
        policy: any chunk-predicting policy.
        mode: one of :data:`MODES`.
        execute_steps: how many actions of a chunk to consume before re-planning.
            Defaults to ``policy.chunk_size`` for ``open_loop`` and ``1``
            otherwise. Setting ``execute_steps < chunk_size`` is the classic
            "predict 50, execute 10" compromise used on real arms.
        decay: recency decay for ``temporal_ens``.
        action_min/max: optional per-dimension safety clamp applied after ensembling.
    """

    def __init__(
        self,
        policy: Policy,
        mode: str = "open_loop",
        execute_steps: int | None = None,
        decay: float = 0.5,
        action_min: np.ndarray | None = None,
        action_max: np.ndarray | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.policy = policy
        self.mode = mode
        self.decay = float(decay)
        default_steps = policy.chunk_size if mode == "open_loop" else 1
        self.execute_steps = int(execute_steps or default_steps)
        self.action_min = None if action_min is None else np.asarray(action_min, np.float32)
        self.action_max = None if action_max is None else np.asarray(action_max, np.float32)
        self._pending: list[np.ndarray] = []
        self._ensemble: dict[int, list[np.ndarray]] = defaultdict(list)
        self._t = 0
        self.stats = ExecutorStats()

    # ------------------------------------------------------------------ lifecycle
    def reset(self, task: str = "") -> None:
        """Start a new episode.

        ``stats`` deliberately keeps accumulating: a benchmark reports inference
        calls per episode *and* over the whole run, and silently zeroing the
        counter on every reset is how those two numbers drift apart.
        """
        self.policy.reset(task)
        self._pending.clear()
        self._ensemble.clear()
        self._t = 0

    # ------------------------------------------------------------------ main loop
    def act(self, obs: Observation) -> np.ndarray:
        if self.mode == "temporal_ens":
            action = self._act_temporal(obs)
        else:
            action = self._act_sequential(obs)
        action = self._clamp(action)
        self._t += 1
        return action

    def _act_sequential(self, obs: Observation) -> np.ndarray:
        if not self._pending:
            chunk = np.asarray(self.policy.predict_chunk(obs), dtype=np.float32)
            if chunk.ndim == 1:
                chunk = chunk[None, :]
            self.stats.n_inference += 1
            self._pending = [a for a in chunk[: self.execute_steps]]
        action = self._pending.pop(0)
        self.stats.n_steps += 1
        return action

    def _act_temporal(self, obs: Observation) -> np.ndarray:
        chunk = np.asarray(self.policy.predict_chunk(obs), dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        self.stats.n_inference += 1
        horizon = min(len(chunk), max(self.execute_steps, 1))
        for i in range(horizon):
            self._ensemble[self._t + i].append(chunk[i])
        preds = self._ensemble.pop(self._t, [])
        if not preds:
            return self._clamp(chunk[0])
        # Newest prediction gets weight 1, older ones decay geometrically.
        weights = np.array([self.decay ** (len(preds) - 1 - j) for j in range(len(preds))], np.float32)
        weights /= weights.sum()
        action = np.tensordot(weights, np.stack(preds), axes=1)
        self.stats.n_steps += 1
        return action.astype(np.float32)

    def _clamp(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if self.action_min is not None:
            action = np.maximum(action, self.action_min)
        if self.action_max is not None:
            action = np.minimum(action, self.action_max)
        return action
