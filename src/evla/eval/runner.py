"""Rollout runner: the only place where a policy touches an environment.

Design choices that make the numbers comparable across policies and machines:

* **Paired seeds.** Episodes are seeded from a fixed list, so every policy under
  comparison faces the *same* initial scenes. This removes scene-to-scene variance
  from the comparison instead of hoping it averages out.
* **Success is read from the environment, never inferred** from reward or final
  distance. Each adapter defines success exactly once.
* **Latency is measured inside the control loop**, not in a separate microbenchmark,
  so it includes observation conversion and normalisation -- the parts that
  surprise people on hardware.
* **Everything is serialised** to JSON, including per-episode records, so a report
  can be regenerated without re-running the GPU.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from evla.control.chunk_executor import ChunkExecutor
from evla.envs.base import EmbodiedEnv
from evla.eval.stats import RateEstimate, wilson_interval
from evla.types import Policy


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    success: bool
    steps: int
    wall_time_s: float
    mean_step_ms: float
    final_distance: float | None = None
    info: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    label: str
    n_episodes: int
    mode: str
    execute_steps: int
    policy: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    episodes: list[EpisodeResult] = field(default_factory=list)
    latency: dict = field(default_factory=dict)
    executor: dict = field(default_factory=dict)
    wall_time_s: float = 0.0
    video: str | None = None

    # ------------------------------------------------------------------ derived
    @property
    def successes(self) -> int:
        return sum(int(e.success) for e in self.episodes)

    @property
    def rate_estimate(self) -> RateEstimate:
        return wilson_interval(self.successes, len(self.episodes))

    @property
    def mean_steps(self) -> float:
        return float(np.mean([e.steps for e in self.episodes])) if self.episodes else 0.0

    @property
    def mean_episode_s(self) -> float:
        return float(np.mean([e.wall_time_s for e in self.episodes])) if self.episodes else 0.0

    @property
    def mean_best_pos_err(self) -> float | None:
        """Mean best block-to-goal distance, when the environment reports it.

        Binary success saturates at 0 for a half-trained policy and hides real
        progress; this graded number does not.
        """
        values = [e.info.get("best_pos_err") for e in self.episodes]
        if not values or any(v is None for v in values):
            return None
        arr = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(arr)):
            return None
        return float(arr.mean())

    @property
    def mean_best_coverage(self) -> float | None:
        """Mean closest-to-goal pose coverage (0..1), when the env reports it."""
        values = [e.info.get("best_coverage") for e in self.episodes]
        if not values or any(v is None for v in values):
            return None
        return float(np.mean(np.asarray(values, dtype=float)))

    @property
    def control_hz(self) -> float:
        """Effective closed-loop control frequency (steps per wall-clock second)."""
        if not self.episodes:
            return 0.0
        return float(1000.0 / max(1e-6, float(np.mean([e.mean_step_ms for e in self.episodes]))))

    def as_dict(self) -> dict:
        est = self.rate_estimate
        return {
            "label": self.label,
            "policy": self.policy,
            "env": self.env,
            "mode": self.mode,
            "execute_steps": self.execute_steps,
            "n_episodes": len(self.episodes),
            "successes": self.successes,
            "success_rate": round(est.rate, 4),
            "ci_lo": round(est.lo, 4),
            "ci_hi": round(est.hi, 4),
            "mean_steps": round(self.mean_steps, 2),
            "mean_episode_s": round(self.mean_episode_s, 2),
            "control_hz": round(self.control_hz, 2),
            "mean_best_pos_err": (None if self.mean_best_pos_err is None
                                  else round(self.mean_best_pos_err, 2)),
            "mean_best_coverage": (None if self.mean_best_coverage is None
                                   else round(self.mean_best_coverage, 4)),
            "latency": self.latency,
            "executor": self.executor,
            "wall_time_s": round(self.wall_time_s, 1),
            "video": self.video,
            "episodes": [asdict(e) for e in self.episodes],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> EvalResult:
        """Load a result JSON written by :meth:`save`.

        Derived fields and free-form annotations (``success_rate``,
        ``optimizations``, ...) are recomputed or ignored rather than trusted, so a
        result file can never disagree with its own per-episode records.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        episodes = [EpisodeResult(**e) for e in raw.pop("episodes")]
        known = {f for f in cls.__dataclass_fields__}
        return cls(episodes=episodes, **{k: v for k, v in raw.items() if k in known})


def _save_video(frames: list[np.ndarray], path: Path, fps: int = 10) -> str | None:
    if not frames:
        return None
    try:
        import imageio.v2 as imageio
    except ImportError:  # pragma: no cover - optional dependency
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    return str(path)


def seed_everything(seed: int) -> None:
    """Seed every RNG that can influence a rollout.

    A flow-matching policy draws its initial noise at inference time, so without
    this an evaluation is not reproducible: the same checkpoint on the same scene
    gives a different success. Seeding also *tightens* paired comparisons, because
    both configurations then start from the same noise as well as the same scene.
    """
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - CPU-only install
        pass


def run_episode(
    env: EmbodiedEnv,
    executor: ChunkExecutor,
    seed: int,
    episode: int = 0,
    record_video: Path | None = None,
    frame_stride: int = 2,
) -> tuple[EpisodeResult, list[np.ndarray]]:
    seed_everything(seed)
    obs = env.reset(seed=seed)
    executor.reset(env.task or "")
    frames: list[np.ndarray] = []
    step_times: list[float] = []
    success, steps, info = False, 0, {}
    t_start = time.perf_counter()
    for step in range(env.max_steps):
        if record_video is not None and step % frame_stride == 0:
            frame = env.render_frame()
            if frame is not None:
                frames.append(np.asarray(frame))
        t0 = time.perf_counter()
        action = executor.act(obs)
        result = env.step(np.asarray(action, np.float32))
        step_times.append((time.perf_counter() - t0) * 1000.0)
        obs = result.obs
        steps = step + 1
        success = bool(result.success)
        info = result.info
        if result.done:
            break
    if record_video is not None:
        frame = env.render_frame()
        if frame is not None:
            frames.append(np.asarray(frame))
    episode_result = EpisodeResult(
        episode=episode,
        seed=seed,
        success=success,
        steps=steps,
        wall_time_s=round(time.perf_counter() - t_start, 3),
        mean_step_ms=round(float(np.mean(step_times)) if step_times else 0.0, 3),
        final_distance=(
            float(info["pos_err"]) if info.get("pos_err") is not None
            else float(info["distance"]) if info.get("distance") is not None
            else None
        ),
        info={k: v for k, v in info.items() if isinstance(v, (int, float, bool, str))},
    )
    return episode_result, frames


def evaluate(
    env_factory: Callable[[], EmbodiedEnv],
    policy: Policy,
    *,
    label: str,
    n_episodes: int = 10,
    seeds: list[int] | None = None,
    mode: str = "open_loop",
    execute_steps: int | None = None,
    decay: float = 0.5,
    video_path: str | Path | None = None,
    progress: bool = True,
) -> EvalResult:
    """Run ``n_episodes`` rollouts and aggregate them into an :class:`EvalResult`."""
    if seeds is None:
        seeds = list(range(n_episodes))
    n_episodes = len(seeds)
    env = env_factory()
    executor = ChunkExecutor(policy, mode=mode, execute_steps=execute_steps, decay=decay)
    episodes: list[EpisodeResult] = []
    video: str | None = None
    t_start = time.perf_counter()
    for i, seed in enumerate(seeds):
        record = Path(video_path) if video_path is not None and i == 0 else None
        result, frames = run_episode(env, executor, seed=seed, episode=i, record_video=record)
        episodes.append(result)
        if record is not None:
            video = _save_video(frames, record)
        if progress:
            print(
                f"  [{label}] episode {i + 1}/{n_episodes} seed={seed} "
                f"success={result.success} steps={result.steps} "
                f"{result.mean_step_ms:.0f} ms/step",
                flush=True,
            )
    env.close()
    latency = {}
    inner = getattr(policy, "stats", None)
    if inner is not None and hasattr(inner, "summary"):
        latency = inner.summary()
    return EvalResult(
        label=label,
        n_episodes=n_episodes,
        mode=mode,
        execute_steps=int(executor.execute_steps),
        policy=policy.describe(),
        env=env.spec(),
        episodes=episodes,
        latency=latency,
        executor=executor.stats.as_dict(),
        wall_time_s=round(time.perf_counter() - t_start, 1),
        video=video,
    )
