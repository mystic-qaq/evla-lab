"""Fixed-rate deployment runtime: HAL -> policy -> safety -> actuators.

The loop is written the way a robot loop has to be written:

* **Deadline scheduling, not sleeping.** Each step has a deadline of
  ``1 / control_hz``; the loop measures how much of the budget was actually used and
  counts misses. That counter -- not the mean latency -- is the number to quote to
  a hardware team.
* **Chunked inference is amortised across steps**, so the arm runs at the control
  rate while the GPU runs one forward pass per chunk.
* **Every action passes the safety layer**, and the layer's counters are reported
  with the run. A policy that "works" only because the safety layer is clamping most
  of its output is not working.
* **Episode bookkeeping lives here**, so the same runtime serves a simulator and an
  arm without conditional code paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from evla.control.chunk_executor import ChunkExecutor
from evla.deploy.hal.base import RobotBackend
from evla.deploy.safety import SafetyConfig, SafetyLayer
from evla.types import Policy


@dataclass
class RuntimeConfig:
    control_hz: float = 10.0
    max_steps: int = 300
    episodes: int = 1
    seed: int = 0
    mode: str = "open_loop"
    execute_steps: int | None = None
    realtime: bool = True
    reset_backend: bool = True

    @classmethod
    def from_dict(cls, cfg: dict | None) -> RuntimeConfig:
        cfg = dict(cfg or {})
        known = set(cls.__dataclass_fields__)
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"unknown runtime keys: {sorted(unknown)}")
        return cls(**cfg)


@dataclass
class RuntimeReport:
    episodes: int = 0
    successes: int = 0
    steps: int = 0
    deadline_misses: int = 0
    budget_used: float = 0.0
    safety: dict = field(default_factory=dict)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_max_ms: float = 0.0
    inference_calls: int = 0
    wall_time_s: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.episodes)

    @property
    def miss_rate(self) -> float:
        return self.deadline_misses / max(1, self.steps)

    def as_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 4),
            "steps": self.steps,
            "inference_calls": self.inference_calls,
            "inference_calls_per_step": round(self.inference_calls / max(1, self.steps), 3),
            "deadline_misses": self.deadline_misses,
            "deadline_miss_rate": round(self.miss_rate, 4),
            "budget_used_mean": round(self.budget_used, 4),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "latency_max_ms": round(self.latency_max_ms, 2),
            "safety": self.safety,
            "wall_time_s": round(self.wall_time_s, 1),
        }


class DeployRuntime:
    def __init__(
        self,
        backend: RobotBackend,
        policy: Policy,
        config: RuntimeConfig | None = None,
        safety: SafetyLayer | None = None,
        safety_config: SafetyConfig | None = None,
        video_path: str | Path | None = None,
    ) -> None:
        self.backend = backend
        self.policy = policy
        self.cfg = config or RuntimeConfig()
        self.safety = safety or SafetyLayer(safety_config, action_dim=backend.spec.action_dim)
        self.executor = ChunkExecutor(
            policy, mode=self.cfg.mode, execute_steps=self.cfg.execute_steps
        )
        self.video_path = Path(video_path) if video_path else None
        self._frames: list[np.ndarray] = []

    # ------------------------------------------------------------------ loop
    def run(self) -> RuntimeReport:
        report = RuntimeReport()
        budget = 1.0 / max(self.cfg.control_hz, 1e-6)
        step_latencies: list[float] = []
        t_start = time.perf_counter()
        for ep in range(self.cfg.episodes):
            seed = self.cfg.seed + ep
            if self.cfg.reset_backend and hasattr(self.backend, "reset_episode"):
                self.backend.reset_episode(seed=seed)  # type: ignore[attr-defined]
            obs = self.backend.read()
            self.executor.reset(getattr(obs, "task", "") or "")
            self.safety.reset_episode()
            self._frames = []
            for _ in range(self.cfg.max_steps):
                t0 = time.perf_counter()
                if self.video_path is not None and len(self._frames) < 400:
                    frame = getattr(self.backend, "render", lambda *_: None)()
                    if frame is not None:
                        self._frames.append(np.asarray(frame))
                raw_action = self.executor.act(obs)
                step_ms = (time.perf_counter() - t0) * 1000.0
                action = self.safety.filter(raw_action, latency_ms=step_ms)
                self.backend.send(action)
                report.steps += 1
                step_latencies.append(step_ms)
                report.budget_used += step_ms / 1000.0 / budget
                if step_ms / 1000.0 > budget:
                    report.deadline_misses += 1
                obs = self.backend.read()
                if getattr(self.backend, "episode_done", False):
                    break
                if self.cfg.realtime:
                    remaining = budget - (time.perf_counter() - t0)
                    if remaining > 0:
                        time.sleep(remaining)
            report.episodes += 1
            report.successes += int(getattr(self.backend, "episode_success", False))
        report.wall_time_s = time.perf_counter() - t_start
        report.safety = self.safety.status.as_dict()
        report.inference_calls = self.executor.stats.n_inference
        if step_latencies:
            arr = np.asarray(step_latencies)
            report.latency_p50_ms = float(np.percentile(arr, 50))
            report.latency_p95_ms = float(np.percentile(arr, 95))
            report.latency_max_ms = float(arr.max())
        report.budget_used /= max(1, report.steps)
        self._save_video()
        return report

    def _save_video(self) -> None:
        if self.video_path is None or not self._frames:
            return
        try:
            import imageio.v2 as imageio

            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(self.video_path, self._frames, fps=int(max(1, self.cfg.control_hz)),
                            macro_block_size=1)
        except ImportError:  # pragma: no cover - optional dependency
            pass
