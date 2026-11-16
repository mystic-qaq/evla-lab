"""Statistics tests.

These are the tests that protect the *claims* in the README, so they check known
values rather than just "it runs".
"""

import math

import pytest

from evla.eval.leaderboard import build_leaderboard, mcnemar_exact
from evla.eval.stats import (
    bootstrap_interval,
    cohens_h,
    required_episodes,
    summarize_episodes,
    two_proportion_ztest,
    wilson_interval,
)


def test_wilson_matches_reference_values():
    # Wilson 95% interval for 15/20 is approximately [0.531, 0.888].
    est = wilson_interval(15, 20)
    assert est.rate == pytest.approx(0.75)
    assert 0.52 < est.lo < 0.55
    assert 0.87 < est.hi < 0.90


def test_wilson_stays_inside_unit_interval_at_boundaries():
    for successes in (0, 20):
        est = wilson_interval(successes, 20)
        assert 0.0 <= est.lo <= est.rate <= est.hi <= 1.0
    assert wilson_interval(0, 20).lo == 0.0
    assert wilson_interval(20, 20).hi == 1.0


def test_wilson_narrows_with_more_episodes():
    narrow = wilson_interval(150, 200)
    wide = wilson_interval(15, 20)
    assert (narrow.hi - narrow.lo) < (wide.hi - wide.lo) / 2


def test_wilson_rejects_bad_counts():
    with pytest.raises(ValueError):
        wilson_interval(3, 0)
    with pytest.raises(ValueError):
        wilson_interval(5, 3)


def test_ztest_detects_difference_and_is_symmetric():
    a = two_proportion_ztest(40, 50, 25, 50)
    b = two_proportion_ztest(25, 50, 40, 50)
    assert a["p_value"] == pytest.approx(b["p_value"])
    assert a["diff"] == pytest.approx(-b["diff"])
    assert a["significant_95"] is True
    assert b["significant_95"] is True


def test_ztest_identical_samples_gives_p_one():
    res = two_proportion_ztest(10, 20, 10, 20)
    assert res["p_value"] == pytest.approx(1.0)
    assert res["significant_95"] is False


def test_required_episodes_grows_as_effect_shrinks():
    big = required_episodes(0.3, 0.8)
    small = required_episodes(0.60, 0.65)
    assert 5 < big < 60
    assert small > big * 5
    assert small < 20000


def test_required_episodes_rejects_equal_rates():
    with pytest.raises(ValueError):
        required_episodes(0.5, 0.5)


def test_bootstrap_interval_brackets_the_mean():
    values = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 5
    lo, hi = bootstrap_interval(values, n_boot=500)
    mean = sum(values) / len(values)
    assert lo <= mean <= hi


def test_cohens_h_bounds():
    assert cohens_h(0.5, 0.5) == pytest.approx(0.0)
    assert cohens_h(0.9, 0.1) > 0.8


def test_summarize_episodes():
    est = summarize_episodes([True] * 7 + [False] * 3)
    assert est.episodes == 10
    assert est.successes == 7
    assert math.isclose(est.rate, 0.7)


def test_mcnemar_exact_known_values():
    # No discordant pairs -> no evidence at all.
    assert mcnemar_exact(0, 0) == 1.0
    # 5 vs 0 discordant pairs: 2 * (1/32) = 0.0625
    assert mcnemar_exact(5, 0) == pytest.approx(0.0625)
    # Symmetric in its arguments.
    assert mcnemar_exact(8, 2) == pytest.approx(mcnemar_exact(2, 8))


def test_mcnemar_p_value_monotone_in_imbalance():
    assert mcnemar_exact(10, 1) < mcnemar_exact(6, 3) < mcnemar_exact(5, 4)


class _FakeResult:
    """Minimal stand-in for EvalResult used by the leaderboard tests."""

    def __init__(self, label, outcomes, hz=10.0):
        self.label = label
        self.episodes = [
            type("E", (), {"seed": i, "success": bool(s)}) for i, s in enumerate(outcomes)
        ]
        self.control_hz = hz

    @property
    def rate_estimate(self):
        return wilson_interval(sum(int(e.success) for e in self.episodes), len(self.episodes))


def test_leaderboard_pairs_by_seed():
    base = _FakeResult("base", [0, 0, 1, 1, 0, 0, 1, 1])
    better = _FakeResult("better", [1, 1, 1, 1, 0, 1, 1, 1])
    board = build_leaderboard([base, better], reference="base")
    assert board["reference"] == "base"
    row = next(r for r in board["rows"] if r["label"] == "better")
    assert row["successes"] == 7
    assert row["wins_over_ref"] == 3   # seeds 0, 1, 5: base failed, challenger succeeded
    assert row["losses_vs_ref"] == 0
    assert row["losses_vs_ref"] == 0
    assert row["delta_vs_ref"] == pytest.approx(7 / 8 - 4 / 8)
    assert board["rows"][0]["label"] == "better"
