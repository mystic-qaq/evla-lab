"""Safety-layer and chunk-executor tests.

Both components are pure logic with no GPU or simulator dependency, which is
exactly why they can be (and are) tested exhaustively. In a real deployment these
are the two pieces that decide whether an arm survives a bad policy output.
"""

import numpy as np
import pytest

from evla.control.chunk_executor import ChunkExecutor
from evla.deploy.safety import SafetyConfig, SafetyLayer
from evla.types import Observation, Policy


class ScriptedPolicy(Policy):
    """Emits a fixed chunk so executor behaviour is fully deterministic."""

    def __init__(self, chunk):
        self._chunk = np.asarray(chunk, np.float32)
        self.chunk_size = self._chunk.shape[0]
        self.action_dim = self._chunk.shape[1]
        self.calls = 0

    def predict_chunk(self, obs):
        self.calls += 1
        return self._chunk


def _obs():
    return Observation(state=np.zeros(2, np.float32), images={}, task="t")


# --------------------------------------------------------------------- executor
def test_open_loop_reuses_actions():
    policy = ScriptedPolicy([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
    ex = ChunkExecutor(policy, mode="open_loop")
    actions = [ex.act(_obs()) for _ in range(4)]
    assert policy.calls == 1
    np.testing.assert_allclose(actions[2], [0.3, 0.3])
    assert ex.stats.n_steps == 4 and ex.stats.n_inference == 1
    assert ex.stats.reuse_ratio == 4.0


def test_open_loop_replans_when_chunk_exhausted():
    policy = ScriptedPolicy([[0.1, 0.1], [0.2, 0.2]])
    ex = ChunkExecutor(policy, mode="open_loop")
    for _ in range(5):
        ex.act(_obs())
    assert policy.calls == 3  # 2 + 2 + 1


def test_execute_steps_decouples_prediction_from_execution():
    policy = ScriptedPolicy(np.tile([[0.5, 0.5]], (50, 1)))
    ex = ChunkExecutor(policy, mode="open_loop", execute_steps=10)
    for _ in range(30):
        ex.act(_obs())
    assert policy.calls == 3
    assert ex.stats.reuse_ratio == 10.0


def test_receding_mode_infers_every_step():
    policy = ScriptedPolicy(np.tile([[0.5, 0.5]], (8, 1)))
    ex = ChunkExecutor(policy, mode="receding")
    for _ in range(6):
        ex.act(_obs())
    assert policy.calls == 6
    assert ex.stats.reuse_ratio == 1.0


def test_temporal_ensemble_averages_overlapping_predictions():
    policy = ScriptedPolicy([[1.0, 1.0]] * 3)
    ex = ChunkExecutor(policy, mode="temporal_ens", execute_steps=3, decay=0.5)
    first = ex.act(_obs())
    second = ex.act(_obs())
    np.testing.assert_allclose(first, [1.0, 1.0])
    # At t=1 two predictions overlap: newest weight 1, previous 0.5.
    np.testing.assert_allclose(second, [1.0, 1.0])
    assert policy.calls == 2


def test_executor_clips_actions_to_limits():
    policy = ScriptedPolicy([[5.0, -5.0]])
    ex = ChunkExecutor(policy, mode="open_loop", action_min=np.array([-1, -1]), action_max=np.array([1, 1]))
    np.testing.assert_allclose(ex.act(_obs()), [1.0, -1.0])


def test_executor_rejects_unknown_mode():
    with pytest.raises(ValueError):
        ChunkExecutor(ScriptedPolicy([[0.0, 0.0]]), mode="magic")


# ----------------------------------------------------------------------- safety
def test_safety_clips_to_limits_without_estopping():
    safety = SafetyLayer(SafetyConfig(action_min=[-1, -1], action_max=[1, 1]), action_dim=2)
    for _ in range(10):
        out = safety.filter(np.array([9.0, -9.0]))
    np.testing.assert_allclose(out, [1.0, -1.0])
    assert safety.status.clipped == 10
    assert safety.estopped is False


def test_safety_rate_limits_delta():
    safety = SafetyLayer(SafetyConfig(max_delta=0.2), action_dim=2)
    safety.filter(np.array([0.0, 0.0]))
    out = safety.filter(np.array([1.0, -1.0]))
    np.testing.assert_allclose(out, [0.2, -0.2])
    assert safety.status.rate_limited == 1


def test_safety_replaces_nonfinite_and_trips_estop():
    safety = SafetyLayer(SafetyConfig(max_consecutive_violations=2), action_dim=2)
    safety.filter(np.array([0.1, 0.1]))
    assert np.all(np.isfinite(safety.filter(np.array([np.nan, 0.0]))))
    safety.filter(np.array([np.inf, np.inf]))
    assert safety.estopped is True
    assert safety.status.nonfinite == 2


def test_safety_estop_is_latched_and_releasable():
    safety = SafetyLayer(SafetyConfig(max_consecutive_violations=1), action_dim=2)
    safety.filter(np.array([np.nan, np.nan]))
    frozen = safety.filter(np.array([0.5, 0.5]))
    np.testing.assert_allclose(frozen, [0.0, 0.0])
    safety.release_estop()
    np.testing.assert_allclose(safety.filter(np.array([0.5, 0.5])), [0.5, 0.5])


def test_safety_holds_position_on_latency_timeout():
    safety = SafetyLayer(SafetyConfig(max_latency_ms=10.0), action_dim=2)
    safety.filter(np.array([0.3, 0.3]))
    out = safety.filter(np.array([0.9, 0.9]), latency_ms=25.0)
    np.testing.assert_allclose(out, [0.3, 0.3])
    assert safety.status.timeouts == 1


def test_safety_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        SafetyConfig.from_dict({"max_deltaa": 1.0})
