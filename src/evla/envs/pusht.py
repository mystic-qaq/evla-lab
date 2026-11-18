"""Adapter for the Push-T manipulation task (gym-pusht / Pymunk).

A simulated pusher moves a T-shaped block onto a fixed target pose. LeRobot's
public demonstrations provide RGB observations, pusher positions and absolute
position commands for this low-cost policy-learning benchmark.

The adapter is deliberately defensive about key names because `gym_pusht` has
changed its observation dict between minor releases.
"""

from __future__ import annotations

import numpy as np

from evla.envs.base import EmbodiedEnv
from evla.types import Observation, StepResult


class PushTEnv(EmbodiedEnv):
    name = "pusht"
    action_dim = 2
    # Must match the dataset's instruction *verbatim*: SmolVLA is
    # language-conditioned, so a reworded instruction is a different task.
    task = "Push the T-shaped block onto the T-shaped target."
    max_steps = 300

    def __init__(self, obs_size: int = 96, max_steps: int = 300) -> None:
        try:
            import gym_pusht  # noqa: F401
            import gymnasium as gym
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PushTEnv requires `pip install -e .[sim]` (gym-pusht + pygame)."
            ) from exc
        self._env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
        self._env = self._env.unwrapped
        self.max_steps = int(max_steps)
        self.obs_size = int(obs_size)
        self._t = 0
        self._success = False
        self._last_frame: np.ndarray | None = None
        self._best_pos_err = float("inf")
        self._max_contacts = 0
        self._best_coverage = 0.0
        self._best_coverage_pos_err = float("nan")
        self._best_coverage_angle_err_deg = float("nan")

    def reset(self, *, seed: int | None = None) -> Observation:
        raw, info = self._env.reset(seed=seed)
        self._t = 0
        self._success = bool(info.get("is_success", False))
        self._best_pos_err = self._pose_error(info)
        self._max_contacts = int(info.get("n_contacts", 0))
        self._best_coverage = float(info.get("coverage", 0.0))
        self._best_coverage_pos_err = self._best_pos_err
        self._best_coverage_angle_err_deg = self._angle_error_deg(info)
        return self._to_obs(raw)

    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        raw, reward, terminated, truncated, info = self._env.step(action)
        self._t += 1
        success = bool(info.get("is_success", getattr(self._env, "_success", False)))
        self._success = success
        pos_err = self._pose_error(info)
        self._best_pos_err = min(self._best_pos_err, pos_err)
        self._max_contacts = max(self._max_contacts, int(info.get("n_contacts", 0)))
        coverage = float(info.get("coverage", 0.0))
        angle_err_deg = self._angle_error_deg(info)
        if coverage > self._best_coverage:
            self._best_coverage = coverage
            self._best_coverage_pos_err = pos_err
            self._best_coverage_angle_err_deg = angle_err_deg
        truncated = bool(truncated) or self._t >= self.max_steps
        return StepResult(
            obs=self._to_obs(raw),
            reward=float(reward),
            done=bool(terminated) or truncated,
            truncated=truncated,
            success=success,
            # Graded signal: success is binary and saturates at 0 on a partially
            # trained policy, while "how close did the block get" keeps showing
            # whether the model is learning anything at all. `max_contacts` feeds the
            # failure taxonomy (never touched the block vs. touched it and missed).
            info={
                "is_success": success,
                # Intersection area divided by target area. Success requires > 0.95.
                "coverage": coverage,
                "best_coverage": self._best_coverage,
                "pos_err": pos_err,
                "angle_err_deg": angle_err_deg,
                "best_pos_err": self._best_pos_err,
                "best_coverage_pos_err": self._best_coverage_pos_err,
                "best_coverage_angle_err_deg": self._best_coverage_angle_err_deg,
                "max_contacts": self._max_contacts,
            },
        )

    @staticmethod
    def _pose_error(info: dict) -> float:
        """Block-to-goal translation error in pixels (NaN if unavailable)."""
        block = info.get("block_pose")
        goal = info.get("goal_pose")
        if block is None or goal is None:
            return float("nan")
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        goal = np.asarray(goal, dtype=np.float32).reshape(-1)
        return float(np.linalg.norm(block[:2] - goal[:2]))

    @staticmethod
    def _angle_error_deg(info: dict) -> float:
        block = info.get("block_pose")
        goal = info.get("goal_pose")
        if block is None or goal is None:
            return float("nan")
        delta = float(block[2]) - float(goal[2])
        return float(np.degrees(abs((delta + np.pi) % (2 * np.pi) - np.pi)))

    def render_frame(self, size: int = 256) -> np.ndarray | None:
        try:
            frame = self._env.render()
        except Exception:  # pragma: no cover - headless fallback
            frame = self._last_frame
        if frame is None:
            return None
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        if size != frame.shape[0]:
            idx = np.linspace(0, frame.shape[0] - 1, size).astype(int)
            frame = frame[np.ix_(idx, idx)]
        return frame

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------ helpers
    def _to_obs(self, raw: dict) -> Observation:
        image = None
        for key in ("observation.image", "observation.images.top", "pixels"):
            if key in raw and raw[key] is not None:
                image = np.asarray(raw[key])
                break
        if image is None:
            image = np.zeros((self.obs_size, self.obs_size, 3), np.uint8)
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        self._last_frame = image
        # gym_pusht names proprioception `agent_pos`; LeRobot datasets call it
        # `observation.state`. Both are the 2D end-effector position.
        for key in ("observation.state", "agent_pos"):
            if key in raw:
                state = np.asarray(raw[key], dtype=np.float32).reshape(-1)
                break
        else:
            raise KeyError(f"no proprioceptive field in observation: {list(raw)}")
        return Observation(state=state, images={"top": image}, task=self.task)
