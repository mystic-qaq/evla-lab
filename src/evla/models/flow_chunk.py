"""A 2M-parameter flow-matching chunk policy.

This exists for two reasons:

1. **It is the same generative formulation as SmolVLA.** SmolVLA's action expert
   is a flow-matching head conditioned on a VLM context. Implementing the head
   from scratch on a toy task makes the mechanics (noise schedule, Euler
   integration, chunk horizon, inference-step/latency trade-off) explicit instead
   of hidden behind a library call.
2. **CI and the quickstart need a model that trains in 40 seconds on a CPU.**
   Nothing about the surrounding pipeline -- chunked execution, profiling,
   deployment server, success-rate statistics -- requires a 450M backbone.

Reference: Lipman et al., *Flow Matching for Generative Modeling* (2023);
Black et al., *π0* (2024) for the action-expert formulation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from evla.types import Observation, Policy


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("The flow policy needs PyTorch: `pip install -e .[vla]`") from exc
    return torch


class FlowNet:
    """Factory helper so ``torch.nn`` is imported lazily."""

    @staticmethod
    def build(state_dim: int, chunk_size: int, action_dim: int, hidden: int = 256, depth: int = 3):
        torch = _require_torch()
        nn = torch.nn

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                in_dim = state_dim + chunk_size * action_dim + 1
                layers: list[nn.Module] = []
                dim = in_dim
                for _ in range(depth):
                    layers += [nn.Linear(dim, hidden), nn.SiLU()]
                    dim = hidden
                layers += [nn.Linear(dim, chunk_size * action_dim)]
                self.net = nn.Sequential(*layers)
                # Velocity predictions should start near zero-mean unit-scale.
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)

            def forward(self, state, x_t, t):
                b = x_t.shape[0]
                inp = torch.cat([state, x_t.reshape(b, -1), t.reshape(b, 1)], dim=-1)
                return self.net(inp).reshape(b, chunk_size, action_dim)

        return _Net()


class FlowChunkPolicy(Policy):
    """Rectified-flow policy over action chunks conditioned on proprioceptive state.

    Args:
        state_dim: proprioceptive dimension.
        action_dim: action dimension.
        chunk_size: number of future actions predicted per call.
        num_inference_steps: Euler steps used at sampling time. Fewer steps = lower
            latency and (for a well-trained model) almost identical actions.
        hidden/depth: network width and depth.
    """

    name = "flow_chunk"

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        chunk_size: int = 8,
        num_inference_steps: int = 8,
        hidden: int = 256,
        depth: int = 3,
        device: str = "cpu",
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
    ) -> None:
        torch = _require_torch()
        self.torch = torch
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.num_inference_steps = int(num_inference_steps)
        self.device = torch.device(device)
        self.net = FlowNet.build(self.state_dim, self.chunk_size, self.action_dim, hidden, depth).to(self.device)
        self.net.eval()
        self.state_mean = np.zeros(self.state_dim, np.float32) if state_mean is None else np.asarray(state_mean, np.float32)
        self.state_std = np.ones(self.state_dim, np.float32) if state_std is None else np.asarray(state_std, np.float32)
        self.action_mean = np.zeros(self.action_dim, np.float32) if action_mean is None else np.asarray(action_mean, np.float32)
        self.action_std = np.ones(self.action_dim, np.float32) if action_std is None else np.asarray(action_std, np.float32)

    # ------------------------------------------------------------------ inference
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        return ((state - self.state_mean) / np.maximum(self.state_std, 1e-6)).astype(np.float32)

    def predict_chunk(self, obs: Observation) -> np.ndarray:
        torch = self.torch
        state = self._normalize_state(np.asarray(obs.state, np.float32))
        with torch.inference_mode():
            s = torch.from_numpy(state).to(self.device).unsqueeze(0)
            x = torch.randn(1, self.chunk_size, self.action_dim, device=self.device)
            steps = max(1, self.num_inference_steps)
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((1,), i * dt, device=self.device)
                x = x + dt * self.net(s, x, t)
            chunk = x.squeeze(0).cpu().numpy()
        return (chunk * self.action_std + self.action_mean).astype(np.float32)

    # ------------------------------------------------------------------ training
    def loss(self, state: np.ndarray, target_chunk: np.ndarray) -> Any:
        """Rectified-flow MSE loss for one batch (numpy in, torch scalar out)."""
        torch = self.torch
        s = torch.from_numpy(self._normalize_state(state)).to(self.device)
        a = torch.from_numpy(((target_chunk - self.action_mean) / np.maximum(self.action_std, 1e-6)).astype(np.float32)).to(self.device)
        noise = torch.randn_like(a)
        t = torch.rand(a.shape[0], device=self.device)
        x_t = (1.0 - t)[:, None, None] * noise + t[:, None, None] * a
        v_target = a - noise
        v_pred = self.net(s, x_t, t)
        return ((v_pred - v_target) ** 2).mean()

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "evla.flow_chunk.v1",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "num_inference_steps": self.num_inference_steps,
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
            "state_dict": {k: v.cpu().numpy() for k, v in self.net.state_dict().items()},
            "extra": extra or {},
        }
        np.savez(path.with_suffix(".npz") if path.suffix != ".npz" else path, **{
            k: v for k, v in payload.items() if k != "state_dict"
        }, **{f"param::{k}": v for k, v in payload["state_dict"].items()})

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> FlowChunkPolicy:
        data = np.load(path if str(path).endswith(".npz") else str(path) + ".npz", allow_pickle=True)
        policy = cls(
            state_dim=int(data["state_dim"]),
            action_dim=int(data["action_dim"]),
            chunk_size=int(data["chunk_size"]),
            num_inference_steps=int(data["num_inference_steps"]),
            device=device,
            state_mean=data["state_mean"],
            state_std=data["state_std"],
            action_mean=data["action_mean"],
            action_std=data["action_std"],
        )
        sd = {k[len("param::") :]: policy.torch.from_numpy(data[k]) for k in data.files if k.startswith("param::")}
        policy.net.load_state_dict(sd)
        policy.net.eval()
        return policy

    def describe(self) -> dict:
        base = super().describe()
        n_params = sum(p.numel() for p in self.net.parameters())
        base.update({"params_m": round(n_params / 1e6, 2), "num_inference_steps": self.num_inference_steps})
        return base


def make_scheduler(step: int, total: int, base_lr: float, warmup: int = 50) -> float:
    """Cosine decay with linear warmup; returns the multiplier for ``base_lr``."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))
