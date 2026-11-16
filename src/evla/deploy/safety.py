"""Safety layer: the code that stands between a neural network and an arm.

A learned policy is a stochastic function with no notion of joint limits, torque
limits or self-collision. On a real arm the runtime therefore never forwards an
action directly. This layer is intentionally boring and fully deterministic, and it
is unit-tested -- which is exactly the property you want in the one component that
can break hardware.

Layers, cheapest first:

1. **E-stop.** Latched; once engaged, no policy output reaches the actuators until
   a human releases it.
2. **Finiteness.** NaN/Inf actions are replaced by the last safe action, then
   counted; too many in a row trips the e-stop.
3. **Position limits.** Workspace/joint bounds, per dimension.
4. **Rate limiting.** ``|a_t - a_{t-1}| <= max_delta``. This is what prevents the
   "policy hallucinates a 40 cm jump" failure from becoming a broken wrist.
5. **Watchdog.** If inference latency exceeds the control period, the runtime holds
   the last action instead of extrapolating into the unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SafetyConfig:
    action_min: list[float] | None = None
    action_max: list[float] | None = None
    max_delta: float | None = None
    max_latency_ms: float = 250.0
    max_consecutive_violations: int = 5
    hold_last_on_timeout: bool = True
    estop_on_repeated_clip: bool = False

    @classmethod
    def from_dict(cls, cfg: dict | None) -> SafetyConfig:
        cfg = dict(cfg or {})
        known = set(cls.__dataclass_fields__)
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"unknown safety keys: {sorted(unknown)}")
        return cls(**cfg)


@dataclass
class SafetyStatus:
    clipped: int = 0
    rate_limited: int = 0
    nonfinite: int = 0
    timeouts: int = 0
    estop_events: int = 0
    violations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "clipped": self.clipped,
            "rate_limited": self.rate_limited,
            "nonfinite": self.nonfinite,
            "timeouts": self.timeouts,
            "estop_events": self.estop_events,
        }


class SafetyLayer:
    def __init__(self, config: SafetyConfig | None = None, action_dim: int = 2) -> None:
        self.cfg = config or SafetyConfig()
        self.action_dim = int(action_dim)
        self._lo = None if self.cfg.action_min is None else np.asarray(self.cfg.action_min, np.float32)
        self._hi = None if self.cfg.action_max is None else np.asarray(self.cfg.action_max, np.float32)
        if self._lo is not None and self._hi is not None and self._lo.shape != self._hi.shape:
            raise ValueError("action_min and action_max must have the same shape")
        self.status = SafetyStatus()
        self._last = np.zeros(self.action_dim, np.float32)
        self._estop = False
        self._consecutive = 0

    # ------------------------------------------------------------------ e-stop
    def engage_estop(self, reason: str = "manual") -> None:
        self._estop = True
        self.status.estop_events += 1
        if reason not in self.status.violations:
            self.status.violations.append(reason)

    def release_estop(self) -> None:
        self._estop = False
        self._consecutive = 0

    @property
    def estopped(self) -> bool:
        return self._estop

    # ------------------------------------------------------------------ filtering
    def filter(self, action: np.ndarray, latency_ms: float | None = None) -> np.ndarray:
        """Clamp, rate-limit and watchdog a raw policy action.

        Clipping and rate limiting are *protective* and only counted: a policy whose
        outputs need clamping is poorly calibrated, not dangerous. The e-stop is
        reserved for conditions where continuing is unsafe -- non-finite actions, or
        an optional repeated-clip escalation.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)[: self.action_dim]
        if self._estop:
            return self._last.copy()

        fatal = False

        if not np.all(np.isfinite(action)):
            self.status.nonfinite += 1
            fatal = True
            action = self._last.copy()

        if latency_ms is not None and latency_ms > self.cfg.max_latency_ms:
            self.status.timeouts += 1
            if self.cfg.hold_last_on_timeout:
                return self._last.copy()

        if self._lo is not None:
            clipped = np.clip(action, self._lo, self._hi)
            if not np.allclose(clipped, action):
                self.status.clipped += 1
                fatal = fatal or self.cfg.estop_on_repeated_clip
            action = clipped

        if self.cfg.max_delta is not None and self.cfg.max_delta > 0:
            delta = action - self._last
            limited = np.clip(delta, -self.cfg.max_delta, self.cfg.max_delta)
            if not np.allclose(limited, delta):
                self.status.rate_limited += 1
                fatal = fatal or self.cfg.estop_on_repeated_clip
            action = self._last + limited

        self._consecutive = self._consecutive + 1 if fatal else 0
        if self._consecutive >= self.cfg.max_consecutive_violations:
            self.engage_estop("consecutive safety violations")
        self._last = action.astype(np.float32)
        return self._last.copy()

    def reset_episode(self) -> None:
        self._last = np.zeros(self.action_dim, np.float32)
        self._consecutive = 0
