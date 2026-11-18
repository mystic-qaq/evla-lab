"""Two planning-time observations for the Push-T SmolVLA experiment."""

from __future__ import annotations


def observation_indices(config) -> list[int]:
    if config.n_obs_steps == 1:
        return [0]
    if config.n_obs_steps == 2:
        return [-config.n_action_steps, 0]
    raise ValueError("evla SmolVLA supports one or two observation frames")


def enable_context(policy) -> None:
    """Map the earlier frame to camera2 and concatenate normalized state histories."""
    if policy.config.n_obs_steps == 1 or getattr(policy, "_evla_context_enabled", False):
        return
    observation_indices(policy.config)
    original_images = policy.prepare_images
    original_state = policy.prepare_state

    def prepare_images(batch):
        images = batch["observation.images.camera1"]
        if images.ndim != 5 or images.shape[1] != 2:
            raise ValueError("two-frame SmolVLA expects image tensors shaped (B, 2, C, H, W)")
        return original_images({
            **batch,
            "observation.images.camera1": images[:, 1],
            "observation.images.camera2": images[:, 0],
        })

    def prepare_state(batch):
        import torch

        states = batch["observation.state"]
        if states.ndim != 3 or states.shape[1] != 2:
            raise ValueError("two-frame SmolVLA expects state tensors shaped (B, 2, D)")
        state = torch.cat((states[:, 1], states[:, 0]), dim=-1)
        return original_state({**batch, "observation.state": state})

    policy.prepare_images = prepare_images
    policy.prepare_state = prepare_state
    policy._evla_context_enabled = True
