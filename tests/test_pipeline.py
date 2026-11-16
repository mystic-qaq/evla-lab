"""End-to-end pipeline tests on the toy task (CPU only, no simulator, no GPU).

These exercise the same code paths the SmolVLA experiments use: demo collection ->
chunk dataset -> flow-matching training -> checkpoint round-trip -> chunked
evaluation -> report generation -> deployment runtime -> policy server.
"""

import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from evla.control.chunk_executor import ChunkExecutor
from evla.data.demos import build_chunks, collect_toy_demos, normalization_stats, save_demos
from evla.deploy.hal.sim import SimBackend
from evla.deploy.runtime import DeployRuntime, RuntimeConfig
from evla.deploy.safety import SafetyConfig
from evla.envs.registry import make_env
from evla.envs.toy import ToyReachEnv
from evla.eval.runner import EvalResult, evaluate
from evla.types import Observation, Policy


# ------------------------------------------------------------------------- envs
def test_toy_env_is_seed_deterministic():
    a, b = ToyReachEnv(), ToyReachEnv()
    oa, ob = a.reset(seed=7), b.reset(seed=7)
    np.testing.assert_allclose(oa.state, ob.state)
    for _ in range(5):
        ra = a.step(np.array([0.2, -0.2]))
        rb = b.step(np.array([0.2, -0.2]))
        np.testing.assert_allclose(ra.obs.state, rb.obs.state)


def test_toy_expert_solves_the_task():
    env = ToyReachEnv(drift=0.0)
    for seed in range(5):
        env.reset(seed=seed)
        result = None
        for _ in range(env.max_steps):
            result = env.step(env.expert_action())
            if result.done:
                break
        assert result is not None and result.success


def test_toy_env_render_shape():
    env = ToyReachEnv()
    env.reset(seed=0)
    frame = env.render_frame(64)
    assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8


def test_env_registry_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_env("definitely_not_an_env")


# ------------------------------------------------------------------- toy pipeline
def _tiny_checkpoint(tmp_path: Path, steps: int = 250) -> Path:
    import torch

    from evla.models.flow_chunk import FlowChunkPolicy

    demos = collect_toy_demos(n_episodes=12, seed=0)
    save_demos(tmp_path / "demos.npz", demos)
    s_mean, s_std = normalization_stats(demos["states"])
    a_mean, a_std = normalization_stats(demos["actions"])
    xs, ys = build_chunks(demos["states"], demos["actions"], demos["episode_ids"], 4)
    policy = FlowChunkPolicy(
        state_dim=xs.shape[1], action_dim=ys.shape[2], chunk_size=4, num_inference_steps=4,
        state_mean=s_mean, state_std=s_std, action_mean=a_mean, action_std=a_std,
    )
    opt = torch.optim.AdamW(policy.net.parameters(), lr=3e-3)
    policy.net.train()
    for _ in range(steps):
        loss = policy.loss(xs, ys)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    policy.net.eval()
    policy.save(tmp_path / "flow")
    return tmp_path / "flow.npz"


def test_flow_policy_roundtrip_and_shapes(tmp_path):
    path = _tiny_checkpoint(tmp_path)
    from evla.models.flow_chunk import FlowChunkPolicy

    policy = FlowChunkPolicy.load(path)
    obs = ToyReachEnv().reset(seed=0)
    chunk = policy.predict_chunk(obs)
    assert chunk.shape == (4, 2)
    assert np.all(np.isfinite(chunk))
    assert policy.describe()["params_m"] > 0


def test_chunks_never_cross_episode_boundaries():
    states = np.zeros((5, 6), np.float32)
    actions = np.arange(10, dtype=np.float32).reshape(5, 2)
    ep_ids = np.array([0, 0, 1, 1, 1])
    _, targets = build_chunks(states, actions, ep_ids, chunk_size=3)
    # Last step of episode 0 must be padded with its own action, not step 1's.
    np.testing.assert_allclose(targets[1, 1], actions[1])
    np.testing.assert_allclose(targets[1, 2], actions[1])
    np.testing.assert_allclose(targets[2, 1], actions[3])


def test_evaluate_produces_result_and_json(tmp_path):
    path = _tiny_checkpoint(tmp_path)
    from evla.models.flow_chunk import FlowChunkPolicy

    policy = FlowChunkPolicy.load(path)
    result = evaluate(
        lambda: make_env("toy_reach"), policy, label="tiny", n_episodes=4,
        seeds=[0, 1, 2, 3], mode="open_loop", progress=False,
    )
    assert isinstance(result, EvalResult)
    assert 0.0 <= result.rate_estimate.rate <= 1.0
    assert result.episodes[0].mean_step_ms >= 0
    out = result.save(tmp_path / "eval.json")
    assert json.loads(out.read_text())["n_episodes"] == 4
    reloaded = EvalResult.load(out)
    assert reloaded.successes == result.successes


def test_report_and_leaderboard_files(tmp_path):
    from evla.eval.report import write_report

    path = _tiny_checkpoint(tmp_path)
    from evla.models.flow_chunk import FlowChunkPolicy

    policy = FlowChunkPolicy.load(path)
    results = [
        evaluate(lambda: make_env("toy_reach"), policy, label=f"cfg{i}", n_episodes=3,
                 seeds=[0, 1, 2], progress=False)
        for i in range(2)
    ]
    report = write_report(results, tmp_path / "report")
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "success" in text.lower()
    assert (tmp_path / "report" / "success_rates.png").exists()
    assert (tmp_path / "report" / "pareto.png").exists()


# ------------------------------------------------------------------- deployment
class TargetSeeker(Policy):
    """Hand-written policy: step toward the target. Used to test the runtime, not a model."""

    chunk_size = 1
    action_dim = 2

    def predict_chunk(self, obs: Observation) -> np.ndarray:
        pos, target = obs.state[:2], obs.state[2:4]
        delta = target - pos
        norm = np.linalg.norm(delta)
        return ((delta / norm) if norm > 1e-6 else np.zeros(2))[None, :].astype(np.float32)


def test_runtime_runs_full_loop_and_reports():
    backend = SimBackend(lambda: make_env("toy_reach", drift=0.0), seed=0)
    backend.connect()
    runtime = DeployRuntime(
        backend, TargetSeeker(), RuntimeConfig(control_hz=200, max_steps=48, episodes=2, realtime=False),
        safety_config=SafetyConfig(max_delta=1.0),
    )
    report = runtime.run()
    backend.disconnect()
    assert report.episodes == 2
    assert report.steps > 0
    assert report.inference_calls == report.steps  # chunk_size 1 -> one call per step
    assert 0.0 <= report.success_rate <= 1.0
    assert "clipped" in report.safety


# ----------------------------------------------------------------------- server
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_policy_server_roundtrip():
    from evla.deploy.server import PolicyServer, RemotePolicy

    port = _free_port()
    server = PolicyServer(TargetSeeker(), host="127.0.0.1", port=port)
    thread = threading.Thread(target=server.serve_forever, kwargs={"announce": False}, daemon=True)
    thread.start()
    for _ in range(50):
        try:
            client = RemotePolicy(host="127.0.0.1", port=port)
            break
        except OSError:
            time.sleep(0.05)
    else:  # pragma: no cover
        server.shutdown()
        pytest.fail("server did not accept connections")

    obs = Observation(state=np.array([0.0, 0.0, 1.0, 0.0], np.float32), images={})
    chunk = client.predict_chunk(obs)
    assert chunk.shape == (1, 2)
    assert chunk[0, 0] > 0.9
    assert client.rtt_ms and client.rtt_ms[0] >= 0
    client.close()
    server.shutdown()


def test_chunk_executor_over_remote_server_amortises_calls():
    from evla.deploy.server import PolicyServer, RemotePolicy

    class Chunk48(Policy):
        chunk_size = 48
        action_dim = 2

        def predict_chunk(self, obs):
            return np.full((48, 2), 0.01, np.float32)

    port = _free_port()
    server = PolicyServer(Chunk48(), host="127.0.0.1", port=port)
    threading.Thread(target=server.serve_forever, kwargs={"announce": False}, daemon=True).start()
    for _ in range(50):
        try:
            client = RemotePolicy(host="127.0.0.1", port=port)
            break
        except OSError:
            time.sleep(0.05)
    ex = ChunkExecutor(client, mode="open_loop")
    obs = Observation(state=np.zeros(2, np.float32), images={})
    for _ in range(48):
        ex.act(obs)
    assert ex.stats.n_inference == 1
    client.close()
    server.shutdown()


def test_failure_taxonomy_buckets():
    from evla.eval.failure import classify_episode, taxonomy

    def ep(success=False, contacts=None, best=None, coverage=None):
        return type("E", (), {"success": success,
                              "info": {"max_contacts": contacts, "best_pos_err": best,
                                       "best_coverage": coverage}})()

    assert classify_episode(ep(True)) == "success"
    assert classify_episode(ep(False, contacts=0, best=300.0)) == "never_contacted"
    assert classify_episode(ep(False, contacts=4, best=12.0)) == "contact_near_miss"
    assert classify_episode(ep(False, contacts=9, best=180.0)) == "contact_missed"
    assert classify_episode(ep(False)) == "failed_unknown"
    assert classify_episode(ep(False, contacts=2, coverage=0.9)) == "contact_near_miss"
    assert classify_episode(ep(False, contacts=2, coverage=0.76)) == "contact_missed"

    class FakeResult:
        label = "cfg"
        episodes = [ep(True), ep(False, 0, 250.0), ep(False, 3, 20.0)]

    tax = taxonomy(FakeResult()).as_dict()
    assert tax["counts"]["success"] == 1
    assert tax["counts"]["never_contacted"] == 1
    assert tax["counts"]["contact_near_miss"] == 1
