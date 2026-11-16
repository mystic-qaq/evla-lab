"""SO-101 / SO-100 follower backend (LeRobot serial protocol).

This is the real-hardware backend. It is included because a deployment story that
stops at a simulator is not a deployment story, but it is **not exercised by CI**:
it requires a calibrated arm on a serial port. Everything it needs from the vendor
SDK is imported lazily, so the rest of the repository stays installable and
testable on a laptop.

Practical notes baked into the code:

* SO-101 reports positions in **degrees** by default. The policy is trained on
  normalised actions, so the conversion happens *here*, once, and is documented --
  not sprinkled through the runtime.
* ``max_relative_target`` is passed to LeRobot so a single bad action cannot sweep
  the arm across the table; this is a second, vendor-level rate limit on top of
  :class:`evla.deploy.safety.SafetyLayer`. Belt and braces is correct for hardware.
* Reading joint state and camera frames is one blocking call, so the runtime must
  keep it off the critical path (see ``deploy/runtime.py``).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from evla.deploy.hal.base import RobotSpec
from evla.types import Observation


class SO101Backend:
    def __init__(
        self,
        port: str,
        robot_id: str = "evla_follower",
        cameras: dict[str, Any] | None = None,
        max_relative_target: float = 5.0,
        action_scale: float = 1.0,
        action_offset: np.ndarray | None = None,
        motor_order: list[str] | None = None,
        use_degrees: bool = True,
        control_hz: float = 10.0,
    ) -> None:
        self.port = port
        self.robot_id = robot_id
        self.cameras = cameras or {}
        self.max_relative_target = max_relative_target
        self.action_scale = float(action_scale)
        self.action_offset = None if action_offset is None else np.asarray(action_offset, np.float32)
        self.motor_order = motor_order
        self.use_degrees = use_degrees
        self._robot = None
        self._connected = False
        self.spec = RobotSpec(name="so101", dof=6, control_hz=control_hz, action_dim=6)

    # ------------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        from lerobot.robots.so_follower.so_follower import SO101Follower

        cfg = SO101FollowerConfig(
            port=self.port,
            id=self.robot_id,
            cameras=self.cameras,
            max_relative_target=self.max_relative_target,
            use_degrees=self.use_degrees,
        )
        self._robot = SO101Follower(cfg)
        self._robot.connect()
        if self.motor_order is None:
            obs = self._robot.get_observation()
            self.motor_order = sorted(k[: -len(".pos")] for k in obs if k.endswith(".pos"))
        self.spec = RobotSpec(
            name="so101",
            dof=len(self.motor_order or []),
            joint_names=tuple(self.motor_order or []),
            control_hz=self.spec.control_hz,
            action_dim=len(self.motor_order or []),
        )
        self._connected = True

    # ------------------------------------------------------------------ HAL
    def read(self) -> Observation:
        if self._robot is None:
            raise RuntimeError("connect() first")
        raw = self._robot.get_observation()
        state = np.asarray([raw[f"{m}.pos"] for m in self.motor_order], np.float32)
        images = {k: np.asarray(v, np.uint8) for k, v in raw.items() if not k.endswith(".pos")}
        return Observation(state=state, images=images, task="")

    def send(self, action: np.ndarray) -> None:
        if self._robot is None:
            raise RuntimeError("connect() first")
        action = np.asarray(action, np.float32).reshape(-1)
        if self.action_offset is not None:
            action = action + self.action_offset
        action = action * self.action_scale
        payload = {f"{m}.pos": float(v) for m, v in zip(self.motor_order or [], action, strict=False)}
        self._robot.send_action(payload)

    def step(self, action: np.ndarray) -> Observation:
        """Send then read; the runtime uses ``send``/``read`` separately, this is a shortcut."""
        self.send(action)
        time.sleep(1.0 / max(1.0, self.spec.control_hz))
        return self.read()

    def disconnect(self) -> None:
        if self._robot is not None:
            try:
                self._robot.disconnect()  # disables torque by default
            except Exception:  # pragma: no cover - hardware dependent
                pass
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected
