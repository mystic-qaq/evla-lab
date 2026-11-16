"""Core interfaces shared by every stage of the pipeline.

The whole repository is built around two tiny abstractions:

* :class:`Observation` -- what a robot policy sees at a control step.
* :class:`Policy` -- something that maps an observation to one or more actions.

Because training, inference optimization, deployment and evaluation all speak
these two types, a policy trained in ``evla.train`` can be benchmarked in
``evla.optimize``, served by ``evla.deploy`` and scored by ``evla.eval`` without
any glue code. Keeping the contract this small is what makes it possible to
swap a 450M SmolVLA for a 2M toy MLP in a unit test.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Observation:
    """A single robot observation.

    Attributes:
        state: proprioceptive vector, shape ``(state_dim,)``, float32.
        images: camera name -> ``(H, W, 3)`` uint8 array (RGB).
        task: natural-language instruction, empty string for task-free envs.
    """

    state: np.ndarray
    images: dict[str, np.ndarray] = field(default_factory=dict)
    task: str = ""

    def __post_init__(self) -> None:
        self.state = np.asarray(self.state, dtype=np.float32).reshape(-1)
        self.images = {k: np.asarray(v) for k, v in self.images.items()}

    @property
    def state_dim(self) -> int:
        return int(self.state.shape[0])

    def copy(self) -> Observation:
        return Observation(
            state=self.state.copy(),
            images={k: v.copy() for k, v in self.images.items()},
            task=self.task,
        )


@dataclass
class StepResult:
    """Result of ``env.step``."""

    obs: Observation
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    success: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class Policy(abc.ABC):
    """Action-chunking interface for robot policies.

    A policy maps one observation to ``(chunk_size, action_dim)`` future actions.
    ``chunk_size == 1`` degenerates to a per-step policy. Wrappers in
    :mod:`evla.optimize` and :mod:`evla.control` rely on this contract to add
    precision casts, compilation, quantization and temporal ensembling.
    """

    name: str = "policy"
    action_dim: int = 2
    chunk_size: int = 1

    def reset(self, task: str = "") -> None:  # noqa: B027 - optional hook
        """Called at the start of every episode. Stateless policies ignore it."""

    @abc.abstractmethod
    def predict_chunk(self, obs: Observation) -> np.ndarray:
        """Return actions of shape ``(chunk_size, action_dim)``."""

    def predict(self, obs: Observation) -> np.ndarray:
        """Return the first action of the predicted chunk, shape ``(action_dim,)``."""
        chunk = self.predict_chunk(obs)
        return np.asarray(chunk, dtype=np.float32)[0]

    def describe(self) -> dict[str, Any]:
        """Metadata surfaced in reports and the leaderboard."""
        return {
            "name": self.name,
            "class": type(self).__name__,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
        }
