"""SmolVLA adapter: from LeRobot checkpoint to the repo-wide ``Policy`` contract.

What this adapter is responsible for, and why each part matters on a real robot:

* **Observation encoding.** The checkpoint was trained on LeRobot feature names
  (``observation.images.camera1``, ``observation.state``, ``task``). The camera
  rename is explicit, because a silently dropped image channel is the single most
  common "my policy suddenly got worse" bug.
* **Normalisation.** Actions are predicted in the dataset's normalised space. The
  pre/post processor pair that LeRobot saves next to the weights is reused
  verbatim instead of re-deriving statistics, so evaluation cannot drift from
  training.
* **Chunked inference.** ``predict_action_chunk`` returns all ``chunk_size``
  actions in one forward pass. Exposing the chunk (instead of LeRobot's internal
  one-action queue) is what lets :class:`evla.control.ChunkExecutor` compare
  receding-horizon, open-loop and temporal-ensembling execution on identical
  weights.
* **Runtime knobs.** ``precision`` (fp32/bf16/fp16) and ``num_flow_steps`` are the
  two levers that move control frequency the most; both are applied here so the
  optimization study changes nothing but the lever under test.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np

from evla.types import Observation, Policy

DEFAULT_RENAME = {"observation.image": "observation.images.camera1"}
DEFAULT_TASK = "Push the T-shaped block onto the T-shaped target."


def _torch():
    import torch

    return torch


class SmolVLAPolicy(Policy):
    """Wrap ``lerobot``'s SmolVLA (450M) as an :class:`~evla.types.Policy`.

    Args:
        checkpoint: local path (or hub id) of a ``pretrained_model`` directory.
        device: torch device string.
        precision: ``fp32`` | ``bf16`` | ``fp16``; implemented with ``torch.autocast``.
        num_flow_steps: Euler steps of the flow-matching action expert. Lower is
            faster and, for a converged model, nearly identical.
        image_key: which observation image feeds ``camera1``.
        state_pad_dim: SmolVLA pads proprioception to 32 dims; kept for reference.
    """

    name = "smolvla"

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        precision: str = "fp32",
        num_flow_steps: int | None = None,
        image_key: str = "top",
        rename: dict[str, str] | None = None,
    ) -> None:
        torch = _torch()
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as _Inner

        self.torch = torch
        self.checkpoint = str(checkpoint)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.policy = _Inner.from_pretrained(self.checkpoint)
        self.policy.to(self.device)
        self.policy.eval()
        self.config = self.policy.config
        self.chunk_size = int(self.config.n_action_steps)
        self.action_dim = int(self.config.output_features["action"].shape[0])
        self.image_key = image_key
        self.rename = dict(rename or DEFAULT_RENAME)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config, pretrained_path=self.checkpoint
        )
        self.precision = "fp32"
        self.set_precision(precision)
        if num_flow_steps is not None:
            self.set_num_flow_steps(num_flow_steps)
        self._task = ""

    # ------------------------------------------------------------------ knobs
    def set_precision(self, precision: str) -> None:
        if precision not in ("fp32", "bf16", "fp16"):
            raise ValueError(f"precision must be fp32/bf16/fp16, got {precision!r}")
        self.precision = precision

    def set_num_flow_steps(self, steps: int) -> None:
        if steps < 1:
            raise ValueError("num_flow_steps must be >= 1")
        self.config.num_steps = int(steps)

    def describe(self) -> dict:
        info = super().describe()
        info.update(
            {
                "checkpoint": Path(self.checkpoint).name,
                "precision": self.precision,
                "num_flow_steps": int(self.config.num_steps),
                "params_m": round(sum(p.numel() for p in self.policy.parameters()) / 1e6, 1),
            }
        )
        return info

    # ------------------------------------------------------------------ policy
    def reset(self, task: str = "") -> None:
        self._task = task or self._task
        self.policy.reset()

    @contextmanager
    def _autocast(self):
        torch = self.torch
        if self.precision == "fp32" or self.device.type == "cpu":
            yield
        else:
            dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                yield

    def _batch(self, obs: Observation) -> dict:
        torch = self.torch
        batch: dict = {}
        img = obs.images.get(self.image_key)
        if img is None:
            raise KeyError(f"observation has no image {self.image_key!r}; got {list(obs.images)}")
        key = self.rename.get("observation.image", "observation.image")
        tensor = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)
        tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
        batch[key] = tensor.to(self.device)
        state = torch.from_numpy(np.asarray(obs.state, np.float32)).unsqueeze(0).to(self.device)
        batch["observation.state"] = state
        batch["task"] = [obs.task or self._task or DEFAULT_TASK]
        return self.preprocessor(batch)

    def predict_chunk(self, obs: Observation) -> np.ndarray:
        torch = self.torch
        batch = self._batch(obs)
        with torch.inference_mode(), self._autocast():
            chunk = self.policy.predict_action_chunk(batch)
        chunk = chunk.squeeze(0).float().cpu()
        chunk = self.postprocessor(chunk)
        chunk = chunk.float().numpy() if hasattr(chunk, "numpy") else np.asarray(chunk, np.float32)
        # `predict_action_chunk` returns what the checkpoint was configured with; trust
        # the tensor over the config so a mismatch cannot silently reshape actions.
        chunk = np.asarray(chunk, np.float32).reshape(-1, self.action_dim)
        self.chunk_size = int(chunk.shape[0])
        return chunk

    # ------------------------------------------------------------------ helpers
    def warmup(self, n: int = 3, obs: Observation | None = None) -> None:
        """Run a few forwards so the first measured step is not dominated by lazy init."""
        if obs is None:
            h = w = 96
            obs = Observation(
                state=np.zeros(2, np.float32),
                images={self.image_key: np.zeros((h, w, 3), np.uint8)},
                task=self._task or DEFAULT_TASK,
            )
        for _ in range(n):
            self.predict_chunk(obs)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()

    def peak_memory_mb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return round(self.torch.cuda.max_memory_allocated(self.device) / 2**20, 1)
