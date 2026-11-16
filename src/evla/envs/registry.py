"""Factories used by the CLI, configs and tests."""

from __future__ import annotations

from typing import Any

from evla.envs.base import EmbodiedEnv

_ENVS = {}


def register_env(name: str):
    def deco(cls):
        _ENVS[name] = cls
        return cls

    return deco


def available_envs() -> list[str]:
    return sorted(_ENVS)


def make_env(cfg: str | dict[str, Any] | None = None, **overrides) -> EmbodiedEnv:
    """Build an environment from a name or a ``{"name": ..., **kwargs}`` dict."""
    if isinstance(cfg, dict):
        spec = dict(cfg)
        name = spec.pop("name", "toy_reach")
    else:
        spec, name = {}, (cfg or "toy_reach")
    spec.update(overrides)
    if name in ("toy", "toy_reach"):
        from evla.envs.toy import ToyReachEnv

        return ToyReachEnv(**spec)
    if name in ("pusht", "push_t"):
        from evla.envs.pusht import PushTEnv

        return PushTEnv(**spec)
    if name in _ENVS:
        return _ENVS[name](**spec)
    raise ValueError(f"unknown env {name!r}; available: toy_reach, pusht, {available_envs()}")
