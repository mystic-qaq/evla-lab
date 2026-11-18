import argparse
import json
import shlex
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evla.envs.pusht import PushTEnv
from evla.models.smolvla_train import reduce_action_loss


def test_finetune_uses_corrected_trainer_and_prints_reproducible_arguments(tmp_path, monkeypatch, capsys):
    from evla import cli

    config = tmp_path / "train.yaml"
    config.write_text(json.dumps({
        "policy": {"base_checkpoint": "checkpoint with spaces", "chunk_size": 16, "n_action_steps": 8},
        "dataset": {"repo_id": "local/pusht", "root": "data/pusht"},
        "train": {"seed": 42},
    }))
    commands = []
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd: commands.append(cmd) or 0)
    args = argparse.Namespace(config=config, trainer=None, dry_run=False)
    assert cli.cmd_finetune(args) == 0
    command = commands[0]
    assert command[:3] == [sys.executable, "-m", "evla.models.smolvla_train"]
    assert "--seed=42" in command
    assert shlex.split(capsys.readouterr().out.replace("\\\n", "")) == command


def test_action_loss_uses_only_real_dimensions_and_valid_steps():
    losses = torch.tensor([[[1., 3., 100.], [100., 100., 100.]],
                           [[4., 6., 100.], [8., 10., 100.]]], requires_grad=True)
    padding = torch.tensor([[False, True], [False, False]])

    loss = reduce_action_loss(losses, action_dim=2, padding=padding, reduction="mean")
    assert loss.item() == pytest.approx(32 / 6)
    torch.testing.assert_close(
        reduce_action_loss(losses, action_dim=2, padding=padding, reduction="none"),
        torch.tensor([2., 7.]),
    )
    loss.backward()
    torch.testing.assert_close(losses.grad[..., 2], torch.zeros(2, 2))
    torch.testing.assert_close(losses.grad[0, 1], torch.zeros(3))


def test_pusht_coverage_starts_fresh_and_tracks_peak_pose():
    raw = {"pixels": np.zeros((2, 2, 3), np.uint8), "agent_pos": [0, 0]}

    class FakeGym:
        def reset(self, seed=None):
            self.seed = seed
            return raw, {"block_pose": [0, 0, 0], "goal_pose": [0, 0, 0]}

        def step(self, action):
            coverage = float(action[0])
            return raw, coverage, False, False, {
                "coverage": coverage, "block_pose": [float(action[1]), 0, -np.pi + 0.1],
                "goal_pose": [0, 0, np.pi - 0.1],
            }

    env = PushTEnv.__new__(PushTEnv)
    env._env = FakeGym()
    env.reset(seed=1)
    env.step(np.array([0.8, 5]))
    later = env.step(np.array([0.4, 1])).info
    assert later["best_coverage"] == pytest.approx(0.8)
    assert later["best_pos_err"] == pytest.approx(0)
    assert later["best_coverage_pos_err"] == pytest.approx(5)
    assert later["best_coverage_angle_err_deg"] == pytest.approx(np.degrees(0.2))

    env.reset(seed=2)
    next_episode = env.step(np.array([0.2, 3])).info
    assert next_episode["best_coverage"] == pytest.approx(0.2)
    assert next_episode["best_coverage_pos_err"] == pytest.approx(3)


def test_two_frame_context_uses_previous_view_and_both_states():
    from evla.models.smolvla_context import enable_context, observation_indices

    policy = SimpleNamespace(
        config=SimpleNamespace(n_obs_steps=2, n_action_steps=8),
        prepare_images=lambda batch: batch,
        prepare_state=lambda batch: batch["observation.state"],
    )
    enable_context(policy)
    enable_context(policy)
    assert observation_indices(policy.config) == [-8, 0]
    batch = {
        "observation.images.camera1": torch.stack((torch.zeros(1, 3, 2, 2), torch.ones(1, 3, 2, 2)), dim=1),
        "observation.state": torch.tensor([[[1., 2.], [3., 4.]]]),
    }
    images = policy.prepare_images(batch)
    assert images["observation.images.camera1"].sum() == 12
    assert images["observation.images.camera2"].sum() == 0
    torch.testing.assert_close(policy.prepare_state(batch), torch.tensor([[3., 4., 1., 2.]]))


def test_smolvla_context_duplicates_initial_observation_and_resets_history():
    from evla.models.smolvla import SmolVLAPolicy
    from evla.types import Observation

    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(n_obs_steps=2)
    policy.torch = torch
    policy.device = torch.device("cpu")
    policy.image_key = "top"
    policy.rename = {"observation.image": "observation.images.camera1"}
    policy.preprocessor = lambda batch: batch
    policy.policy = SimpleNamespace(reset=lambda: None, predict_action_chunk=lambda batch: torch.zeros(1, 16, 2))
    policy.postprocessor = lambda chunk: chunk
    policy.precision = "fp32"
    policy.action_dim = 2
    policy._previous_obs = None
    policy._task = ""
    a = Observation(state=np.array([1., 2.]), images={"top": np.zeros((2, 2, 3), np.uint8)})
    b = Observation(state=np.array([3., 4.]), images={"top": np.full((2, 2, 3), 255, np.uint8)})
    first = policy._batch(a)
    torch.testing.assert_close(first["observation.state"], torch.tensor([[[1., 2.], [1., 2.]]]))
    assert policy._previous_obs is None
    policy.predict_chunk(a)
    second = policy._batch(b)
    torch.testing.assert_close(second["observation.state"], torch.tensor([[[1., 2.], [3., 4.]]]))
    assert second["observation.images.camera1"].shape == (1, 2, 3, 2, 2)
    assert second["observation.images.camera1"][:, 0].sum() == 0
    assert second["observation.images.camera1"][:, 1].sum() == 12
    def fail(batch):
        raise RuntimeError("inference failed")

    policy.policy.predict_action_chunk = fail
    with pytest.raises(RuntimeError, match="inference failed"):
        policy.predict_chunk(b)
    np.testing.assert_array_equal(policy._previous_obs.state, a.state)
    policy.reset()
    reset = policy._batch(b)
    torch.testing.assert_close(reset["observation.state"], torch.tensor([[[3., 4.], [3., 4.]]]))
    policy.config.n_obs_steps = 1
    single = policy._batch(b)
    assert single["observation.images.camera1"].shape == (1, 3, 2, 2)
    torch.testing.assert_close(single["observation.state"], torch.tensor([[3., 4.]]))

