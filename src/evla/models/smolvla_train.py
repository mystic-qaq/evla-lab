"""LeRobot's trainer with SmolVLA's valid-action loss on LeRobot 0.4.4."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def reduce_action_loss(losses: Tensor, action_dim: int, padding: Tensor | None, reduction: str) -> Tensor:
    """Reduce flow-matching errors over the physical action dimensions and valid steps."""
    losses = losses[..., :action_dim]
    if reduction not in ("none", "mean"):
        raise ValueError(f"unsupported loss reduction {reduction!r}")
    if padding is None:
        return losses.mean(dim=(1, 2)) if reduction == "none" else losses.mean()

    valid = ~padding.bool()
    losses = losses.masked_fill(~valid.unsqueeze(-1), 0)
    if reduction == "none":
        return losses.sum(dim=(1, 2)) / (valid.sum(dim=1) * action_dim).clamp_min(1)
    return losses.sum() / (valid.sum() * action_dim).clamp_min(1)


def _forward(self, batch, noise=None, time=None, reduction="mean"):
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    from evla.models.smolvla_context import enable_context

    enable_context(self)
    if self.config.adapt_to_pi_aloha:
        batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
        batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

    images, image_masks = self.prepare_images(batch)
    state = self.prepare_state(batch)
    actions = self.prepare_action(batch)
    losses = self.model(
        images, image_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
        state, actions, noise, time,
    )
    loss = reduce_action_loss(losses, batch[ACTION].shape[-1], batch.get("action_is_pad"), reduction)
    return loss, {"loss": loss.detach().mean().item()}


def main() -> None:
    installed = version("lerobot")
    if installed != "0.4.4":
        raise RuntimeError(f"The evla loss adapter targets LeRobot 0.4.4, found {installed}. "
                           "Use --trainer lerobot-train with an upstream release that fixes the loss.")

    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.scripts.lerobot_train import main as train

    from evla.models.smolvla_context import observation_indices

    SmolVLAConfig.observation_delta_indices = property(observation_indices)
    SmolVLAPolicy.forward = _forward
    train()


if __name__ == "__main__":
    main()
