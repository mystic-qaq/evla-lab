"""Turn evaluation JSONs into a report a human (or an interviewer) can read.

Three figures, because they answer three different questions:

1. *Did fine-tuning work?*  success rate with Wilson intervals -- intervals, not
   bare percentages, because 18/20 and 22/25 are not the same evidence.
2. *How fast is the loop?*  success rate against effective control frequency --
   the Pareto front that decides whether an optimization is worth deploying.
3. *Where does the time go?*  per-inference latency distribution, because p95,
   not the mean, decides whether the robot stalls.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from evla.eval.runner import EvalResult

_COLOURS = ["#2f6f9f", "#c0504d", "#4f9d69", "#8064a2", "#e0a33e", "#4bacc6", "#7f7f7f", "#9d4f4f"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def markdown_table(results: list[EvalResult]) -> str:
    graded = any(r.mean_best_coverage is not None for r in results)
    graded_legacy = not graded and any(r.mean_best_pos_err is not None for r in results)
    header = (
        "| # | config | mode (exec/chunk) | success | 95% Wilson CI | control Hz | p50 lat (ms) | "
        "p95 lat (ms) | steps/inference | mean steps |"
        + (" best coverage |" if graded else " best Δpos (px) |" if graded_legacy else "")
        + "\n"
        "|---|--------|-------------------|---------|---------------|------------|--------------|"
        "--------------|-----------------|------------|"
        + ("-----------------|" if (graded or graded_legacy) else "")
        + "\n"
    )
    rows = []
    for i, r in enumerate(results, 1):
        lat = r.latency or {}
        reuse = r.executor.get("action_reuse", "-")
        row = (
            f"| {i} | {r.label} | {r.mode} ({r.execute_steps}/{r.policy.get('chunk_size', '?')}) | "
            f"**{r.rate_estimate.rate:.0%}** | {r.rate_estimate.lo:.0%}-{r.rate_estimate.hi:.0%} | "
            f"{r.control_hz:.1f} | {lat.get('p50_ms', '-')} | {lat.get('p95_ms', '-')} | "
            f"{reuse} | {r.mean_steps:.0f} |"
        )
        if graded:
            cov = r.mean_best_coverage
            row += f" {'-' if cov is None else f'{cov:.2f}'} |"
        elif graded_legacy:
            value = r.mean_best_pos_err
            row += f" {'-' if value is None else f'{value:.0f}'} |"
        rows.append(row)
    note = (
        "\n`control Hz` is offline simulation throughput computed from mean environment step time. "
        "`p50/p95 lat` measure individual inference calls with CUDA synchronisation. "
        "`steps/inference` counts the environment steps served by one prediction. "
        "`best coverage` averages the maximum coverage of each episode.\n"
    )
    return header + "\n".join(rows) + "\n" + note


def plot_success(results: list[EvalResult], path: Path, title: str = "Task success rate") -> Path | None:
    if not results:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(max(5.0, 0.62 * len(results) + 2.2), 3.2))
    for i, r in enumerate(results):
        est = r.rate_estimate
        ax.bar(i, est.rate, color=_COLOURS[i % len(_COLOURS)], alpha=0.85, width=0.66)
        ax.errorbar(
            i,
            est.rate,
            yerr=[[est.rate - est.lo], [est.hi - est.rate]],
            fmt="none",
            ecolor="#333333",
            capsize=4,
            lw=1.1,
        )
        ax.text(i, est.hi + 0.03, f"{est.successes}/{est.episodes}", ha="center", fontsize=8, color="#333")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([r.label for r in results], rotation=25, ha="right")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _short_labels(results: list[EvalResult]) -> dict[str, str]:
    """Strip the shared prefix so annotations/legends stay readable."""
    labels = [r.label for r in results]
    if len(labels) < 2:
        return {lab: lab for lab in labels}
    prefix = labels[0]
    for lab in labels[1:]:
        while not lab.startswith(prefix) and prefix:
            prefix = prefix[:-1]
    cut = prefix.rfind("__") + 2 if "__" in prefix else len(prefix)
    out = {}
    for lab in labels:
        short = lab[cut:] if cut and lab.startswith(lab[:cut]) else lab
        out[lab] = short.replace("__", " · ") or lab
    return out


def plot_pareto(results: list[EvalResult], path: Path) -> Path | None:
    """Quality vs effective control frequency.

    The y-axis is the graded coverage when the environment reports it: binary success
    is usually 0 or saturated, which collapses this figure into a single row of points
    at y=0 and hides the trade-off the figure exists to show.
    """
    if not results:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    hz_values = [max(1e-3, r.control_hz) for r in results]
    graded = [r.mean_best_coverage for r in results]
    use_coverage = all(v is not None for v in graded)
    short = _short_labels(results)
    for i, (r, hz) in enumerate(zip(results, hz_values, strict=False)):
        est = r.rate_estimate
        if use_coverage:
            y, yerr = float(graded[i]), None  # type: ignore[arg-type]
        else:
            y = est.rate
            yerr = [[max(0.0, est.rate - est.lo)], [max(0.0, est.hi - est.rate)]]
        ax.errorbar(
            hz, y, yerr=yerr, fmt=_MARKERS[i % len(_MARKERS)], ms=9,
            color=_COLOURS[i % len(_COLOURS)], markeredgecolor="white",
            ecolor="#999999", elinewidth=1.0, capsize=3, zorder=3,
            label=f"{short[r.label]}  ({est.successes}/{est.episodes})",
        )
    if max(hz_values) / min(hz_values) > 20:
        ax.set_xscale("log")
    ax.margins(x=0.20, y=0.18)
    if use_coverage:
        ax.axhline(0.95, color="#bbbbbb", ls="--", lw=0.9)
        ax.text(min(hz_values), 0.955, "success threshold", fontsize=7, color="#777")
        ax.set_ylabel("best pose coverage (graded)")
    else:
        ax.axhline(1.0, color="#bbbbbb", ls="--", lw=0.8)
        ax.set_ylabel("success rate (Wilson 95% CI)")
    ax.set_xlabel("effective control frequency (Hz)")
    ax.set_ylim(0, 1.05)

    ax.grid(alpha=0.25, which="both")
    ax.set_title("Quality / control-rate Pareto frontier")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_coverage(results: list[EvalResult], path: Path) -> Path | None:
    """Graded quality per config -- the figure that still discriminates when success is 0."""
    rows = [(r, r.mean_best_coverage) for r in results if r.mean_best_coverage is not None]
    if not rows:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(max(5.2, 0.62 * len(rows) + 2.2), 3.3))
    for i, (_r, cov) in enumerate(rows):
        ax.bar(i, cov, color=_COLOURS[i % len(_COLOURS)], alpha=0.85, width=0.66)
        ax.text(i, cov + 0.015, f"{cov:.2f}", ha="center", fontsize=8, color="#333")
    ax.axhline(0.95, color="#c0504d", ls="--", lw=0.9)
    ax.text(len(rows) - 0.5, 0.962, "success threshold = 0.95", fontsize=7,
            color="#c0504d", ha="right")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r.label for r, _ in rows], rotation=25, ha="right")
    ax.set_ylabel("mean best pose coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("How close did each configuration get?")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_latency(results: list[EvalResult], path: Path) -> Path | None:
    rows = [(r.label, r.episodes) for r in results if r.episodes]
    if not rows:
        return None
    plt = _mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for i, (label, eps) in enumerate(rows):
        values = [e.mean_step_ms for e in eps]
        ax1.plot(range(len(values)), values, marker="o", ms=3, lw=1.2, label=label,
                 color=_COLOURS[i % len(_COLOURS)])
    ax1.set_xlabel("episode")
    ax1.set_ylabel("ms per control step")
    ax1.set_title("Control-loop step cost")
    ax1.legend(fontsize=7, frameon=False)
    ax1.grid(alpha=0.25)

    labels = [label for label, _ in rows]
    means = [sum(e.mean_step_ms for e in eps) / len(eps) for _, eps in rows]
    ax2.barh(range(len(rows)), means, color=[_COLOURS[i % len(_COLOURS)] for i in range(len(rows))], alpha=0.85)
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("mean ms per step")
    ax2.set_title("Mean step cost")
    ax2.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _step_from_label(label: str) -> int | None:
    """Extract a step budget from labels like ``expertonly__8k`` or ``run-12000``."""
    match = re.search(r"(\d+(?:\.\d+)?)k\b", label, re.I)
    if match:
        return int(float(match.group(1)) * 1000)
    match = re.search(r"[_-](\d{4,6})$", label)
    if match:
        return int(match.group(1))
    return None


def plot_learning_curve(results: list[EvalResult], path: Path) -> Path | None:
    """How quality evolves with fine-tuning steps, when labels encode a step budget.

    Plots the graded coverage when available: binary success stays at 0 for most of
    the curve (the task needs a 0.95 pose match), so a success-rate learning curve
    would be a flat line at zero and hide all the actual learning.
    """
    points = []
    for r in results:
        step = _step_from_label(r.label)
        if step is not None:
            est = r.rate_estimate
            points.append((step, r, est))
    if len(points) < 3:
        return None
    points.sort(key=lambda p: p[0])
    zero = [r for r in results if _step_from_label(r.label) is None]
    use_coverage = all(r.mean_best_coverage is not None for _, r, _ in points)

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    steps = [p[0] for p in points]
    if use_coverage:
        ys = [float(p[1].mean_best_coverage) for p in points]  # type: ignore[arg-type]
        ax.plot(steps, ys, marker="o", color=_COLOURS[0], lw=1.6, label="fine-tuned (coverage)")
        for step, r, est in points:
            cov = float(r.mean_best_coverage)  # type: ignore[arg-type]
            ax.annotate(f"{cov:.2f}\n{est.successes}/{est.episodes}", (step, cov),
                        textcoords="offset points", xytext=(0, 9), fontsize=7, ha="center")
        for z in zero:
            if z.mean_best_coverage is not None:
                ax.axhline(z.mean_best_coverage, color=_COLOURS[1], ls="--", lw=1.1,
                           label=f"{z.label} ({z.mean_best_coverage:.2f})")
        ax.axhline(0.95, color="#c0504d", ls=":", lw=1.0)
        ax.text(steps[0], 0.955, "success threshold 0.95", fontsize=7, color="#c0504d")
        ax.set_ylabel("best pose coverage (graded)")
        ax.set_ylim(0, 1.06)
    else:
        rates = [p[2].rate for p in points]
        lows = [p[2].lo for p in points]
        highs = [p[2].hi for p in points]
        ax.plot(steps, rates, marker="o", color=_COLOURS[0], lw=1.6, label="fine-tuned")
        ax.fill_between(steps, lows, highs, color=_COLOURS[0], alpha=0.18, lw=0)
        for z in zero:
            ax.axhline(z.rate_estimate.rate, color=_COLOURS[1], ls="--", lw=1.2, label=z.label)
        for step, _, est in points:
            ax.annotate(f"{est.successes}/{est.episodes}", (step, est.rate),
                        textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center")
        ax.set_ylabel("success rate")
        ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel("fine-tuning steps")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("Does more fine-tuning keep helping?")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _taxonomy_markdown(results: list[EvalResult]) -> str | None:
    """Include the failure taxonomy when the environments report enough signal."""
    from evla.eval.failure import taxonomy, to_markdown

    has_signal = any(
        e.info.get("max_contacts") is not None or e.info.get("best_pos_err") is not None
        for r in results
        for e in r.episodes
    )
    if not has_signal:
        return None
    return to_markdown([taxonomy(r) for r in results], title=None)


def load_results(paths: list[str | Path]) -> list[EvalResult]:
    """Load result JSONs from files or directories (aggregate files are skipped)."""
    out = []
    for path in paths:
        path = Path(path)
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for f in files:
            raw = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or "episodes" not in raw:
                continue  # summary.json / leaderboard.json live in the same folder
            out.append(EvalResult.load(f))
    return out


def write_report(results: list[EvalResult], out_dir: str | Path, title: str = "Evaluation report") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "success": plot_success(results, out_dir / "success_rates.png"),
        "coverage": plot_coverage(results, out_dir / "coverage.png"),
        "pareto": plot_pareto(results, out_dir / "pareto.png"),
        "latency": plot_latency(results, out_dir / "latency.png"),
        "learning_curve": plot_learning_curve(results, out_dir / "learning_curve.png"),
    }
    lines = [f"# {title}", "", markdown_table(results), "## Figures", ""]
    for name, path in figures.items():
        if path is not None:
            lines += [f"### {name}", "", f"![{name}]({path.name})", ""]
    tax = _taxonomy_markdown(results)
    if tax:
        lines += ["## Failure taxonomy", "", tax, ""]
    lines += ["## Raw", "", "Per-episode records and full configuration are in the sibling JSON files.", ""]
    report = out_dir / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps([r.as_dict() for r in results], indent=2), encoding="utf-8"
    )
    return report
