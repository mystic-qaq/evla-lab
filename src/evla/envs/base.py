"""Environment abstraction used by training, evaluation and the deployment sim backend.

Only the pieces the pipeline needs are required: reset/step, a success flag and a
human-viewable frame. Adapters for concrete simulators live next to this file.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np

from evla.types import Observation, StepResult


class EmbodiedEnv(abc.ABC):
    """Minimal Gym-like interface with an explicit task-success signal."""

    name: str = "env"
    action_dim: int = 2
    max_steps: int = 64
    task: str = ""

    def __init__(self) -> None:
        self._t = 0

    @abc.abstractmethod
    def reset(self, *, seed: int | None = None) -> Observation:
        """Reset the scene and return the first observation."""

    @abc.abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        """Apply ``action`` of shape ``(action_dim,)``."""

    def render_frame(self, size: int = 256) -> np.ndarray | None:
        """Optional RGB frame for videos. Return ``None`` if unsupported."""
        return None

    def close(self) -> None:  # noqa: B027 - optional hook, not every env owns resources
        """Release simulator resources."""

    @property
    def episode_step(self) -> int:
        return self._t

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "action_dim": self.action_dim,
            "max_steps": self.max_steps,
        }


def obs_from_dict(raw: dict[str, Any], task: str = "") -> Observation:
    """Convert a gymnasium-style dict observation into :class:`Observation`."""
    state = np.asarray(raw["observation.state"], dtype=np.float32).reshape(-1)
    images = {}
    for key, value in raw.items():
        if not key.startswith("observation.images.") or value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 4:  # (1, H, W, C) from a batched simulator
            arr = arr[0]
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        images[key.replace("observation.images.", "")] = arr
    return Observation(state=state, images=images, task=task)
