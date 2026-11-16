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

_COLOURS = ["#2f6f9f", "#c0504d", "#4f9d69", "#8064a2", "#e0a33e", "#4bacc6", "#7f7f7f"]


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def markdown_table(results: list[EvalResult]) -> str:
    graded = any(r.mean_best_pos_err is not None for r in results)
    header = (
        "| # | config | mode (exec/chunk) | success | 95% Wilson CI | control Hz | p50 lat (ms) | "
        "p95 lat (ms) | steps/inference | mean steps |"
        + (" best Δpos (px) |" if graded else "")
        + "\n"
        "|---|--------|-------------------|---------|---------------|------------|--------------|"
        "--------------|-----------------|------------|"
        + ("-----------------|" if graded else "")
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
            value = r.mean_best_pos_err
            row += f" {'-' if value is None else f'{value:.0f}'} |"
        rows.append(row)
    note = (
        "\n`control Hz` is the effective closed-loop rate (1 / mean environment step time, "
        "including reused actions); `p50/p95 lat` are per-inference latencies measured with "
        "CUDA synchronisation; `steps/inference` is how many control steps one forward pass buys."
        " `best Δpos` is the mean closest block-to-goal distance reached during an episode -- a"
        " graded signal that keeps moving when binary success is still stuck at zero.\n"
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


def plot_pareto(results: list[EvalResult], path: Path) -> Path | None:
    if not results:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    hz_values = [max(1e-3, r.control_hz) for r in results]
    for i, (r, hz) in enumerate(zip(results, hz_values, strict=False)):
        est = r.rate_estimate
        ax.errorbar(
            hz,
            est.rate,
            yerr=[[max(0.0, est.rate - est.lo)], [max(0.0, est.hi - est.rate)]],
            fmt="o",
            ms=8,
            color=_COLOURS[i % len(_COLOURS)],
            markeredgecolor="white",
            ecolor="#999999",
            elinewidth=1.0,
            capsize=3,
            zorder=3,
            label=r.label,
        )
        ax.annotate(
            r.label,
            (hz, est.rate),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=7,
        )
    if max(hz_values) / min(hz_values) > 20:
        ax.set_xscale("log")
    ax.margins(x=0.15, y=0.15)
    ax.axhline(1.0, color="#bbbbbb", ls="--", lw=0.8)
    ax.set_xlabel("effective control frequency (Hz)")
    ax.set_ylabel("success rate (Wilson 95% CI)")
    ax.set_ylim(-0.08, 1.12)
    ax.grid(alpha=0.25, which="both")
    ax.set_title("Success / latency Pareto frontier")
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
    """Success rate vs training steps, when the labels encode a step budget."""
    points = []
    for r in results:
        step = _step_from_label(r.label)
        if step is not None:
            est = r.rate_estimate
            points.append((step, est.rate, est.lo, est.hi, r.label))
    if len(points) < 3:
        return None
    points.sort()
    zero = [r for r in results if _step_from_label(r.label) is None]
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    steps = [p[0] for p in points]
    rates = [p[1] for p in points]
    lows = [p[2] for p in points]
    highs = [p[3] for p in points]
    ax.plot(steps, rates, marker="o", color=_COLOURS[0], lw=1.6, label="fine-tuned")
    ax.fill_between(steps, lows, highs, color=_COLOURS[0], alpha=0.18, lw=0)
    for z in zero:
        ax.axhline(z.rate_estimate.rate, color=_COLOURS[1], ls="--", lw=1.2, label=z.label)
    for step, rate, _, _, label in points:
        ax.annotate(label.split("__")[-1], (step, rate), textcoords="offset points", xytext=(0, 8),
                    fontsize=7, ha="center")
    ax.set_xlabel("fine-tuning steps")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
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
    return to_markdown([taxonomy(r) for r in results], title="Failure taxonomy")


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
