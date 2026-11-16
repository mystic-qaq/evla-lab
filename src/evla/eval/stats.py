"""Success-rate statistics.

A manipulation benchmark reports success rates out of 20-50 episodes. At that
sample size a single episode is worth 2-5 percentage points, so ranking teams by
raw success rate without an interval is close to ranking them by coin flips. This
module provides the minimum honesty kit:

* :func:`wilson_interval` -- the interval to quote for a single policy. Wilson is
  preferred over the textbook Wald interval because it stays inside [0, 1] and
  behaves at the 0% / 100% boundaries that small benchmarks constantly hit.
* :func:`two_proportion_ztest` -- "is team A actually better than team B?".
* :func:`required_episodes` -- how many rollouts are needed *before* the
  experiment, so a 5-point difference is detectable at all.
* :func:`bootstrap_interval` -- distribution-free interval for derived metrics
  (e.g. success per second, or success under a latency budget).

No SciPy dependency: the normal CDF is one ``erfc`` call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Z_95 = 1.959963984540054


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, |err| < 1e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


@dataclass
class RateEstimate:
    successes: int
    episodes: int
    rate: float
    lo: float
    hi: float
    method: str = "wilson"

    def as_dict(self) -> dict:
        return {
            "successes": self.successes,
            "episodes": self.episodes,
            "rate": round(self.rate, 4),
            "ci_lo": round(self.lo, 4),
            "ci_hi": round(self.hi, 4),
            "ci_method": self.method,
        }

    def __str__(self) -> str:
        return f"{self.rate:.1%} [{self.lo:.1%}, {self.hi:.1%}] ({self.successes}/{self.episodes})"


def wilson_interval(successes: int, episodes: int, z: float = Z_95) -> RateEstimate:
    """Wilson score interval for a binomial proportion."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0 <= successes <= episodes:
        raise ValueError("successes must be within [0, episodes]")
    p = successes / episodes
    denom = 1.0 + z * z / episodes
    centre = (p + z * z / (2 * episodes)) / denom
    half = (z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes * episodes))) / denom
    return RateEstimate(successes, episodes, p, max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_ztest(s1: int, n1: int, s2: int, n2: int) -> dict:
    """Two-sided test of ``p1 == p2`` using the pooled normal approximation."""
    if min(n1, n2) <= 0:
        raise ValueError("sample sizes must be positive")
    p1, p2 = s1 / n1, s2 / n2
    pooled = (s1 + s2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"diff": 0.0, "z": 0.0, "p_value": 1.0, "significant_95": False}
    z = (p1 - p2) / se
    p_value = 2 * (1 - _phi(abs(z)))
    return {
        "rate_a": round(p1, 4),
        "rate_b": round(p2, 4),
        "diff": round(p1 - p2, 4),
        "z": round(z, 3),
        "p_value": round(p_value, 4),
        "significant_95": bool(p_value < 0.05),
    }


def required_episodes(
    p_a: float,
    p_b: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> int:
    """Episodes **per arm** needed to detect ``|p_a - p_b|`` with the given power.

    Normal approximation, the same one used to size A/B tests. It is the number a
    team should compute *before* running the tournament: distinguishing 60% from
    70% at 80% power already needs ~330 episodes per policy.
    """
    if p_a == p_b:
        raise ValueError("p_a and p_b must differ")
    z_alpha = _ppf(1 - alpha / 2) if two_sided else _ppf(1 - alpha)
    z_beta = _ppf(power)
    var = p_a * (1 - p_a) + p_b * (1 - p_b)
    n = ((z_alpha + z_beta) ** 2 * var) / ((p_a - p_b) ** 2)
    return int(math.ceil(n))


def bootstrap_interval(
    values: np.ndarray,
    stat=np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for an arbitrary statistic over episodes."""
    values = np.asarray(values, np.float64).reshape(-1)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    stats = np.array([stat(values[i]) for i in idx])
    return (float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2))))


def cohens_h(p_a: float, p_b: float) -> float:
    """Effect size for two proportions; |h| < 0.2 is small, > 0.8 large."""
    return 2 * math.asin(math.sqrt(p_a)) - 2 * math.asin(math.sqrt(p_b))


def summarize_episodes(successes: list[bool]) -> RateEstimate:
    return wilson_interval(int(sum(successes)), len(successes))
