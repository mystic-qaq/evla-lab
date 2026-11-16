"""Hardware abstraction layer.

Everything the deployment runtime needs from a robot is three methods. That is
deliberate: it means the *same* control loop, safety layer and policy server run
against MuJoCo in CI and against a 6-DoF arm on a desk. It is also what makes the
repository reviewable -- a reader can check the runtime without owning hardware.

Backends are expected to be *slow to construct and honest about units*: joint
positions in the vendor's units (SO-101 uses degrees by default, MuJoCo uses
normalised coordinates), because silently converting units at the HAL boundary is
how policies end up commanding 100x too large actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from evla.types import Observation


@dataclass
class RobotSpec:
    name: str = "generic"
    dof: int = 6
    joint_names: tuple[str, ...] = ()
    control_hz: float = 10.0
    action_dim: int = 6


@runtime_checkable
class RobotBackend(Protocol):
    """Minimal read/act interface implemented by every backend."""

    spec: RobotSpec

    def connect(self) -> None:
        """Open the connection / reset the simulator."""

    def read(self) -> Observation:
        """Return the current observation. Must be non-blocking on hardware."""

    def send(self, action: np.ndarray) -> None:
        """Command one action (already clipped by the safety layer)."""

    def disconnect(self) -> None:
        """Release the hardware safely (torque off / close sim)."""

    def is_connected(self) -> bool:
        ...
