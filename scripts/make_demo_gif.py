"""Render README assets from the toy environment (no simulator, no GPU, no training).

Produces:

* ``assets/toy_reach.gif``   -- the scripted expert solving the task
* ``assets/chunking_ab.gif`` -- identical policy, identical seed, receding vs open-loop

The second asset is the point: same weights, same scene, different execution
strategy. Against a fast-moving target the open-loop arm commits to a stale plan and
misses, while the receding arm re-aims every step.

    python scripts/make_demo_gif.py --out assets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evla.control.chunk_executor import ChunkExecutor
from evla.envs.toy import ToyReachEnv
from evla.types import Observation, Policy


class FrozenChunkExpert(Policy):
    """Plan = the scripted expert's current action, repeated for the whole chunk.

    Deliberately myopic: the plan is *correct at planning time* and goes stale
    afterwards, which isolates staleness as the only difference between the two
    panels.
    """

    action_dim = 2

    def __init__(self, chunk_size: int = 8) -> None:
        self.chunk_size = int(chunk_size)

    def predict_chunk(self, obs: Observation) -> np.ndarray:
        pos, target = obs.state[:2], obs.state[2:4]
        delta = target - pos
        norm = float(np.linalg.norm(delta))
        direction = delta / norm if norm > 1e-6 else np.zeros(2, np.float32)
        return np.tile(direction, (self.chunk_size, 1)).astype(np.float32)


def _rollout(mode: str, execute_steps: int, drift: float, seed: int) -> tuple[list[np.ndarray], bool]:
    policy = FrozenChunkExpert(chunk_size=8)
    executor = ChunkExecutor(policy, mode=mode, execute_steps=execute_steps)
    env = ToyReachEnv(drift=drift)
    obs = env.reset(seed=seed)
    executor.reset(env.task)
    frames = [env.render_frame(160)]
    success = False
    for _ in range(env.max_steps):
        result = env.step(executor.act(obs))
        obs, success = result.obs, result.success
        frames.append(env.render_frame(160))
        if result.done:
            break
    env.close()
    return frames, success


def _label(frame: np.ndarray, left: str, right: str) -> np.ndarray:
    """Burn the panel captions in, so the figure is self-explanatory on GitHub."""
    try:
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        width = frame.shape[1] // 2
        for x, text in ((4, left), (width + 4, right)):
            draw.rectangle([x - 2, 0, x + 7 * len(text), 11], fill=(255, 255, 255))
            draw.text((x, 1), text, fill=(30, 30, 30))
        return np.asarray(img)
    except ImportError:  # pragma: no cover - PIL ships with torchvision
        return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="assets")
    parser.add_argument("--drift", type=float, default=0.03)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import imageio.v2 as imageio

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. expert demonstration
    env = ToyReachEnv(drift=0.0)
    obs = env.reset(seed=3)
    frames = [env.render_frame(160)]
    for _ in range(env.max_steps):
        result = env.step(env.expert_action())
        obs = result.obs
        frames.append(env.render_frame(160))
        if result.done:
            break
    env.close()
    imageio.mimsave(out / "toy_reach.gif", frames, fps=args.fps, loop=0)
    print(f"wrote {out / 'toy_reach.gif'} ({len(frames)} frames)")

    # 2. receding vs open-loop, same seed
    left, ok_left = _rollout("receding", 1, args.drift, seed=args.seed)
    right, ok_right = _rollout("open_loop", 8, args.drift, seed=args.seed)
    n = min(len(left), len(right))
    composite = [
        _label(
            np.hstack([left[i], right[i]]),
            "receding: re-plan every step",
            "open-loop: 8 steps blind",
        )
        for i in range(n)
    ]
    imageio.mimsave(out / "chunking_ab.gif", composite, fps=args.fps, loop=0)
    print(f"wrote {out / 'chunking_ab.gif'}  receding_success={ok_left}  open_loop_success={ok_right}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
