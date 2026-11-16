"""Latency instrumentation for robot policies.

Every number in the optimization study comes from here, so the measurement
protocol is deliberately explicit:

* CUDA is synchronised before *and* after each call -- an unsynchronised timer
  measures kernel-launch overhead, not inference (a classic 3-10x lie).
* warmup calls are discarded (cuDNN autotune, lazy init, `torch.compile` guards).
* percentiles, not just the mean: a policy whose p95 exceeds the control period
  will stall the robot even if its mean looks fine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from evla.types import Observation, Policy


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:  # pragma: no cover - CPU-only install
        pass


@dataclass
class LatencyStats:
    samples_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        if not self.samples_ms:
            return {"n": 0}
        arr = np.asarray(self.samples_ms, np.float64)
        return {
            "n": int(arr.size),
            "mean_ms": round(float(arr.mean()), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "max_ms": round(float(arr.max()), 2),
            "achiev_hz_at_p95": round(1000.0 / max(1e-6, float(np.percentile(arr, 95))), 2),
        }


class InstrumentedPolicy(Policy):
    """Transparent wrapper that records wall-clock latency of ``predict_chunk``."""

    def __init__(self, inner: Policy, warmup: int = 3) -> None:
        self.inner = inner
        self.name = f"{inner.name}+profiled"
        self.action_dim = inner.action_dim
        self.chunk_size = inner.chunk_size
        self.stats = LatencyStats()
        if warmup:
            self._warmup(warmup)

    def _warmup(self, n: int) -> None:
        try:
            self.inner.warmup(n)  # type: ignore[attr-defined]
            return
        except AttributeError:
            pass
        try:
            obs = self._dummy_obs()
            for _ in range(n):
                self.inner.predict_chunk(obs)
            _sync()
        except Exception:  # pragma: no cover - best effort only
            pass

    def _dummy_obs(self) -> Observation:
        # A zero observation is enough: warmup only exercises shapes and kernels.
        img = np.zeros((96, 96, 3), np.uint8)
        dim = int(getattr(self.inner, "state_dim", 2))
        return Observation(state=np.zeros(dim, np.float32), images={"top": img}, task="")

    def reset(self, task: str = "") -> None:
        self.inner.reset(task)

    def predict_chunk(self, obs: Observation) -> np.ndarray:
        _sync()
        t0 = time.perf_counter()
        chunk = self.inner.predict_chunk(obs)
        _sync()
        self.stats.samples_ms.append((time.perf_counter() - t0) * 1000.0)
        return chunk

    def describe(self) -> dict:
        info = self.inner.describe()
        info["latency"] = self.stats.summary()
        return info
