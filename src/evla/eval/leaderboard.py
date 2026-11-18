"""Leaderboard with *paired* significance testing.

Because :func:`evla.eval.runner.evaluate` uses the same seed list for every policy,
the episode outcomes form matched pairs. That enables a much sharper test than
comparing two independent proportions: McNemar's test on the discordant pairs.

* ``b`` = reference succeeded, challenger failed
* ``c`` = reference failed, challenger succeeded

If a change is cosmetic, ``b`` and ``c`` are roughly equal and the exact binomial
p-value is large. This is the difference between "we got 5 points more" and "we
got 5 points more, and on the same 20 scenes that was 5 wins vs 0 losses
(p = 0.06)". For a competition that ranks teams by success rate, that distinction
is the whole argument.
"""

from __future__ import annotations

import math

from evla.eval.runner import EvalResult


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2.0 * tail))


def paired_counts(reference: EvalResult, other: EvalResult) -> tuple[int, int, int]:
    """Return ``(both_ok, ref_only, other_only)`` over shared seeds."""
    ref = {e.seed: e.success for e in reference.episodes}
    counts = {"both": 0, "ref_only": 0, "other_only": 0}
    for e in other.episodes:
        if e.seed not in ref:
            continue
        if ref[e.seed] and e.success:
            counts["both"] += 1
        elif ref[e.seed]:
            counts["ref_only"] += 1
        elif e.success:
            counts["other_only"] += 1
    return counts["both"], counts["ref_only"], counts["other_only"]


def build_leaderboard(results: list[EvalResult], reference: str | None = None) -> dict:
    """Rank configs and compare each against a reference with a paired McNemar test.

    When no reference is given, the best-performing config is used -- defaulting to
    the first result alphabetically makes every delta meaningless.
    """
    if not results:
        return {"rows": [], "reference": None}
    if reference is None:
        ref = max(results, key=lambda r: r.rate_estimate.rate)
    else:
        ref = next((r for r in results if r.label == reference), None)
        if ref is None:
            raise ValueError(f"reference {reference!r} not among {[r.label for r in results]}")
    rows = []
    for r in sorted(results, key=lambda x: -x.rate_estimate.rate):
        est = r.rate_estimate
        both, b, c = paired_counts(ref, r)
        p = 1.0 if r is ref else mcnemar_exact(b, c)
        rows.append(
            {
                "label": r.label,
                "success_rate": round(est.rate, 4),
                "ci": [round(est.lo, 4), round(est.hi, 4)],
                "successes": est.successes,
                "episodes": est.episodes,
                "control_hz": round(r.control_hz, 2),
                "delta_vs_ref": round(est.rate - ref.rate_estimate.rate, 4),
                "wins_over_ref": c,
                "losses_vs_ref": b,
                "mcnemar_p": p,
                "significant_95": bool(p < 0.05),
            }
        )
    return {
        "reference": ref.label,
        "rows": rows,
        "note": "The paired McNemar test uses shared seeds. Delta is the absolute success-rate change.",
    }


def to_markdown(board: dict, title: str = "Leaderboard") -> str:
    lines = [
        f"# {title}",
        "",
        f"The reference configuration is **{board['reference']}**.",
        "",
        "| rank | config | success | 95% CI | Δ vs ref | wins/losses | McNemar p | control Hz |",
        "|------|--------|---------|--------|----------|-------------|-----------|------------|",
    ]
    for i, row in enumerate(board["rows"], 1):
        sig = " *" if row["significant_95"] else ""
        lines.append(
            f"| {i} | {row['label']} | {row['success_rate']:.1%} | "
            f"{row['ci'][0]:.0%}-{row['ci'][1]:.0%} | {row['delta_vs_ref']:+.1%} | "
            f"{row['wins_over_ref']}/{row['losses_vs_ref']} | {row['mcnemar_p']:.3g}{sig} | "
            f"{row['control_hz']:.1f} |"
        )
    lines += ["", "* marks p < 0.05.", "", board["note"], ""]
    return "\n".join(lines)
