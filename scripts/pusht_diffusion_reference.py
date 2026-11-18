"""Evaluate the published Push-T Diffusion Policy or record its successful rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import draccus
import numpy as np
import torch
from safetensors.torch import load_file

from evla.envs.pusht import PushTEnv
from evla.eval.stats import wilson_interval


class DiffusionReference:
    """Load the published weights and their legacy embedded normalization stats."""

    def __init__(self, checkpoint: Path, steps: int, scheduler: str, device: str) -> None:
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        self.device = torch.device(device)
        raw_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        if raw_config.pop("type") != "diffusion":
            raise ValueError("expected a Diffusion Policy checkpoint")
        raw_config.update(device=device, num_inference_steps=steps, noise_scheduler_type=scheduler)
        config = draccus.decode(DiffusionConfig, raw_config)
        self.policy = DiffusionPolicy(config).eval()
        weights = load_file(checkpoint / "model.safetensors")
        self.policy.load_state_dict({k: v for k, v in weights.items() if k.startswith("diffusion.")})
        self.policy.to(self.device)
        self.action_steps = config.n_action_steps
        if config.n_obs_steps != 2:
            raise ValueError("the published Push-T checkpoint must use two observations")
        stats = {key: value.to(self.device) for key, value in weights.items() if not key.startswith("diffusion.")}
        self.image_mean = stats["normalize_inputs.buffer_observation_image.mean"].reshape(1, 1, 3, 1, 1)
        self.image_std = stats["normalize_inputs.buffer_observation_image.std"].reshape(1, 1, 3, 1, 1)
        self.state_min = stats["normalize_inputs.buffer_observation_state.min"]
        self.state_max = stats["normalize_inputs.buffer_observation_state.max"]
        self.action_min = stats["unnormalize_outputs.buffer_action.min"]
        self.action_max = stats["unnormalize_outputs.buffer_action.max"]

    @torch.inference_mode()
    def predict(self, previous, current) -> np.ndarray:
        images = np.array([[a.images["top"], b.images["top"]] for a, b in zip(previous, current, strict=True)])
        states = np.array([[a.state, b.state] for a, b in zip(previous, current, strict=True)])
        images = torch.as_tensor(images, device=self.device).permute(0, 1, 4, 2, 3).float() / 255
        images = (images - self.image_mean) / (self.image_std + 1e-8)
        states = torch.as_tensor(states, device=self.device)
        states = 2 * (states - self.state_min) / (self.state_max - self.state_min + 1e-8) - 1
        actions = self.policy.diffusion.generate_actions({
            "observation.images": images.unsqueeze(2), "observation.state": states,
        })
        actions = (actions + 1) * (self.action_max - self.action_min) / 2 + self.action_min
        return actions.cpu().numpy()


def run_batch(model: DiffusionReference, seeds: list[int], record: bool):
    envs = [PushTEnv() for _ in seeds]
    current = [env.reset(seed=seed) for env, seed in zip(envs, seeds, strict=True)]
    previous = current.copy()
    last = {}
    frames: list[list[dict]] = [[] for _ in seeds]
    active = list(range(len(seeds)))
    try:
        while active:
            actions = model.predict([previous[i] for i in active], [current[i] for i in active])
            completed = []
            for i, chunk in zip(active, actions, strict=True):
                for action in chunk:
                    action = np.clip(action, 0, 512)
                    result = envs[i].step(action)
                    if record:
                        frames[i].append({
                            "observation.image": current[i].images["top"],
                            "observation.state": current[i].state,
                            "action": action,
                            "next.reward": np.array([result.reward], dtype=np.float32),
                            "next.done": np.array([result.done], dtype=bool),
                            "next.success": np.array([result.success], dtype=bool),
                            "task": current[i].task,
                        })
                    previous[i], current[i] = current[i], result.obs
                    if result.done:
                        last[i] = {"seed": seeds[i], "success": result.success,
                                   "best_coverage": result.info["best_coverage"], "steps": envs[i]._t}
                        completed.append(i)
                        break
            active = [i for i in active if i not in completed]
        return [last[i] for i in range(len(seeds))], frames
    finally:
        for env in envs:
            env.close()


def make_dataset(source: Path, target: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = json.loads((source / "meta/info.json").read_text())["features"]
    features = {k: v for k, v in features.items() if k in (
        "observation.image", "observation.state", "action", "next.reward", "next.done", "next.success",
    )}
    for feature in features.values():
        feature["shape"] = tuple(feature["shape"])
    features["observation.image"].pop("video_info", None)
    return LeRobotDataset.create(
        repo_id="local/pusht-diffusion-success", fps=10, features=features, root=target,
        use_videos=True, image_writer_threads=4, encoder_threads=2, video_backend="pyav",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--scheduler", choices=("DDPM", "DDIM"), default="DDPM")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--record", type=Path, help="write the successful rollouts to this LeRobot dataset")
    parser.add_argument("--dataset-template", type=Path, default=Path("data/pusht"))
    args = parser.parse_args()
    torch.manual_seed(args.seed0)
    torch.set_float32_matmul_precision("high")
    model = DiffusionReference(args.checkpoint, args.steps, args.scheduler, args.device)
    dataset = make_dataset(args.dataset_template, args.record) if args.record else None
    episodes = []
    for seed in range(args.seed0, args.seed0 + args.episodes, args.batch_size):
        seeds = list(range(seed, min(seed + args.batch_size, args.seed0 + args.episodes)))
        batch, frames = run_batch(model, seeds, record=dataset is not None)
        episodes.extend(batch)
        if dataset is not None:
            for row, trajectory in zip(batch, frames, strict=True):
                if row["success"]:
                    row["dataset_episode"] = dataset.num_episodes
                    for frame in trajectory:
                        dataset.add_frame(frame)
                    dataset.save_episode()
        print(f"completed {len(episodes)}/{args.episodes}, successes={sum(row['success'] for row in episodes)}",
              flush=True)
    if dataset is not None:
        dataset.finalize()
    rate = wilson_interval(sum(row["success"] for row in episodes), len(episodes))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "scheduler": args.scheduler, "denoise_steps": args.steps, "batch_size": args.batch_size,
        "successes": rate.successes, "success_rate": rate.rate, "ci_lo": rate.lo, "ci_hi": rate.hi,
        "mean_best_coverage": float(np.mean([row["best_coverage"] for row in episodes])),
        "episodes": episodes,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
