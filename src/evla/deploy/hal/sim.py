"""Simulator backend -- runs the full deployment stack without hardware."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from evla.deploy.hal.base import RobotSpec
from evla.envs.base import EmbodiedEnv
from evla.types import Observation


class SimBackend:
    """Drive a simulator through the robot HAL.

    ``read`` returns the last observation (cached), so the HAL contract
    "reading is cheap and idempotent" holds exactly as it does for an encoder on a
    real arm. ``send`` advances the simulation and latches episode status.
    """

    def __init__(self, env_factory: Callable[[], EmbodiedEnv], seed: int = 0) -> None:
        self._env_factory = env_factory
        self._env: EmbodiedEnv | None = None
        self._seed = seed
        self._connected = False
        self._obs: Observation | None = None
        self.episode_done = False
        self.episode_success = False
        self.episode_step = 0
        self.spec = RobotSpec(name="sim", dof=2, joint_names=("x", "y"), control_hz=10.0, action_dim=2)

    # ------------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        self._env = self._env_factory()
        self.spec = RobotSpec(
            name=f"sim:{self._env.name}",
            dof=self._env.action_dim,
            joint_names=tuple(f"j{i}" for i in range(self._env.action_dim)),
            control_hz=10.0,
            action_dim=self._env.action_dim,
        )
        self.reset_episode(seed=self._seed)
        self._connected = True

    def reset_episode(self, seed: int | None = None) -> Observation:
        assert self._env is not None
        self._seed = self._seed if seed is None else seed
        self._obs = self._env.reset(seed=self._seed)
        self.episode_done = False
        self.episode_success = False
        self.episode_step = 0
        return self._obs

    # ------------------------------------------------------------------ HAL
    def read(self) -> Observation:
        if self._obs is None:
            raise RuntimeError("call connect() before read()")
        return self._obs

    def send(self, action: np.ndarray) -> None:
        assert self._env is not None
        result = self._env.step(np.asarray(action, np.float32))
        self._obs = result.obs
        self.episode_step += 1
        self.episode_done = bool(result.done)
        self.episode_success = bool(result.success)

    def render(self, size: int = 256) -> np.ndarray | None:
        return None if self._env is None else self._env.render_frame(size)

    def disconnect(self) -> None:
        if self._env is not None:
            self._env.close()
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected
