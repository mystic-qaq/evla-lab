"""Demonstration collection and action-chunk dataset construction.

Two ideas worth keeping when moving to a real robot:

* demonstrations are collected with **behavioural noise** (a variable-speed
  scripted controller plus action noise). A policy fit to a perfect oracle learns
  a brittle single trajectory; covariate shift at deployment is what actually
  breaks imitation policies.
* chunks are cut **inside episode boundaries** and padded by repeating the last
  action, never by wrapping into the next episode. A silent off-by-one here is the
  classic reason chunked policies look fine in training loss and fail on hardware.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evla.envs.toy import ToyReachEnv


def collect_toy_demos(
    n_episodes: int = 160,
    seed: int = 0,
    drift: float = 0.012,
    action_noise: float = 0.08,
    max_steps: int = 48,
) -> dict[str, np.ndarray]:
    """Roll out the scripted expert and return stacked transitions."""
    env = ToyReachEnv(drift=drift, max_steps=max_steps)
    states, actions, ep_ids = [], [], []
    successes = 0
    rng = np.random.default_rng(seed)
    for ep in range(n_episodes):
        obs = env.reset(seed=seed * 10_000 + ep)
        speed = rng.uniform(0.85, 1.0)  # per-demo style variation
        done = False
        while not done:
            a = env.expert_action() * speed + rng.normal(0.0, action_noise, size=2).astype(np.float32)
            states.append(obs.state.copy())
            actions.append(np.clip(a, -1.0, 1.0).astype(np.float32))
            ep_ids.append(ep)
            res = env.step(a)
            obs, done = res.obs, res.done
            successes += int(res.success)
        successes = max(successes, 0)
    env.close()
    return {
        "states": np.stack(states).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "episode_ids": np.asarray(ep_ids, np.int64),
    }


def build_chunks(
    states: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair every timestep with the next ``chunk_size`` actions, clipped per episode."""
    n = len(actions)
    targets = np.zeros((n, chunk_size, actions.shape[1]), np.float32)
    for i in range(n):
        ep = episode_ids[i]
        for j in range(chunk_size):
            k = i + j
            if k >= n or episode_ids[k] != ep:
                targets[i, j] = targets[i, j - 1] if j > 0 else actions[i]
            else:
                targets[i, j] = actions[k]
    return states, targets


def normalization_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    return mean, np.maximum(std, 1e-3).astype(np.float32)


def save_demos(path: str | Path, demos: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **demos)
