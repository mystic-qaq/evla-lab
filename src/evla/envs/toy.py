"""A dependency-free 2D reaching task.

Why a toy environment lives in a serious repository:

* It makes the *entire* pipeline (data collection -> training -> chunked
  inference -> deployment server -> evaluation report) runnable on a CPU in a
  few seconds, which is what CI needs. A 450M VLA cannot be exercised in CI.
* It isolates one experimental question that matters on real hardware --
  **how action-chunk length trades control latency against reactivity** -- because
  the target drifts, so an open-loop chunk goes stale. Real arms, conveyor
  belts and human co-workers do the same thing to a policy.

The observation contains both a low-dimensional state and a rendered 96x96 RGB
image, so image-conditioned policies can be exercised with the same interface as
their MuJoCo counterparts.
"""

from __future__ import annotations

import numpy as np

from evla.envs.base import EmbodiedEnv
from evla.types import Observation, StepResult

_OBS_SIZE = 96


class ToyReachEnv(EmbodiedEnv):
    """Push a point mass onto a slowly drifting target.

    Args:
        drift: target speed in world units per step. ``0`` gives a static task.
        noise: std of observation noise added to the proprioceptive state.
        max_steps: episode horizon.
        action_scale: maximum end-effector displacement per control step.
        success_radius: distance under which the task counts as solved.
    """

    name = "toy_reach"
    action_dim = 2
    task = "reach the red target"

    def __init__(
        self,
        drift: float = 0.012,
        noise: float = 0.004,
        max_steps: int = 48,
        action_scale: float = 0.10,
        success_radius: float = 0.07,
    ) -> None:
        super().__init__()
        self.drift = float(drift)
        self.noise = float(noise)
        self.max_steps = int(max_steps)
        self.action_scale = float(action_scale)
        self.success_radius = float(success_radius)
        self._rng = np.random.default_rng(0)
        self._pos = np.zeros(2, dtype=np.float32)
        self._target = np.zeros(2, dtype=np.float32)
        self._target_vel = np.zeros(2, dtype=np.float32)
        self._success = False

    # ------------------------------------------------------------------ env API
    def reset(self, *, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._pos = self._rng.uniform(-0.8, -0.4, size=2).astype(np.float32)
        self._target = self._rng.uniform(0.3, 0.8, size=2).astype(np.float32)
        angle = self._rng.uniform(0.0, 2.0 * np.pi)
        self._target_vel = (self.drift * np.array([np.cos(angle), np.sin(angle)])).astype(np.float32)
        self._success = False
        return self._observe()

    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        action = np.clip(action, -1.0, 1.0) * self.action_scale
        self._pos = np.clip(self._pos + action, -1.0, 1.0)
        self._target = np.clip(self._target + self._target_vel, -0.95, 0.95)
        # Bounce the drifting target off the workspace walls.
        for axis in range(2):
            if self._target[axis] <= -0.95 or self._target[axis] >= 0.95:
                self._target_vel[axis] *= -1.0
        self._t += 1

        dist = float(np.linalg.norm(self._pos - self._target))
        if dist < self.success_radius:
            self._success = True
        reward = -dist
        truncated = self._t >= self.max_steps
        return StepResult(
            obs=self._observe(),
            reward=reward,
            done=self._success or truncated,
            truncated=truncated,
            success=self._success,
            info={"distance": dist, "pos": self._pos.copy(), "target": self._target.copy()},
        )

    # ---------------------------------------------------------------- rendering
    def render_frame(self, size: int = 256) -> np.ndarray:
        frame = np.full((_OBS_SIZE, _OBS_SIZE, 3), 245, dtype=np.uint8)
        # workspace border
        frame[0, :] = frame[-1, :] = 180
        frame[:, 0] = frame[:, -1] = 180
        # faint grid
        for g in (24, 48, 72):
            frame[g, 1:-1] = 232
            frame[1:-1, g] = 232

        def to_px(p: np.ndarray) -> tuple[int, int]:
            x = int((p[0] + 1.0) * 0.5 * (_OBS_SIZE - 1))
            y = int((p[1] + 1.0) * 0.5 * (_OBS_SIZE - 1))
            return x, y

        tx, ty = to_px(self._target)
        px, py = to_px(self._pos)
        rr = 7
        ys, xs = np.ogrid[-rr : rr + 1, -rr : rr + 1]
        disc = xs * xs + ys * ys <= rr * rr
        for cx, cy, colour in ((tx, ty, np.array([220, 60, 60], np.uint8)), (px, py, np.array([40, 90, 200], np.uint8))):
            x0, x1 = max(0, cx - rr), min(_OBS_SIZE, cx + rr + 1)
            y0, y1 = max(0, cy - rr), min(_OBS_SIZE, cy + rr + 1)
            sub = disc[rr - (cy - y0) : rr + (y1 - cy), rr - (cx - x0) : rr + (x1 - cx)]
            frame[y0:y1, x0:x1][sub] = colour
        if size != _OBS_SIZE:
            idx = (np.linspace(0, _OBS_SIZE - 1, size)).astype(int)
            frame = frame[np.ix_(idx, idx)]
        return frame

    # ------------------------------------------------------------------ helpers
    def _observe(self) -> Observation:
        # Target velocity is part of the observation: a tracking system on a real
        # arm reports it, and without it an action chunk is unidentifiable.
        state = np.concatenate([self._pos, self._target, self._target_vel])
        state = state + self._rng.normal(0.0, self.noise, size=6).astype(np.float32)
        return Observation(
            state=state.astype(np.float32),
            images={"top": self.render_frame()},
            task=self.task,
        )

    def expert_action(self) -> np.ndarray:
        """Scripted controller used to collect demonstrations."""
        delta = self._target - self._pos
        norm = np.linalg.norm(delta) + 1e-8
        return (delta / max(norm, self.success_radius)).astype(np.float32)
