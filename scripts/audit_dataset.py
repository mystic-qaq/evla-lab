"""Audit the demonstration set: where it came from and how good the demos really are.

    python scripts/audit_dataset.py --root data/pusht

Any imitation-learning number invites three questions -- where did the data come from,
what does one sample actually contain, and how well do the demonstrations themselves
score on the metric the policy is judged by. The third one is routinely skipped, and on
Push-T it is the interesting one.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

SUCCESS_THRESHOLD = 0.95


def _load(root: Path) -> tuple[pd.DataFrame, dict]:
    shards = sorted(glob.glob(str(root / "data" / "chunk-*" / "*.parquet")))
    if not shards:
        raise SystemExit(f"no parquet shards under {root}/data -- pass the dataset root")
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True), info


def _task_string(root: Path) -> str | None:
    path = root / "meta" / "tasks.parquet"
    if not path.exists():
        return None
    table = pd.read_parquet(path)
    return str(table.index[0]) if len(table) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/pusht", help="dataset root containing meta/ and data/")
    args = parser.parse_args()
    root = Path(args.root)

    frames, info = _load(root)
    episodes = frames.groupby("episode_index")

    print("== provenance ==")
    print(f"episodes        {info['total_episodes']}")
    print(f"frames          {info['total_frames']}  ({len(frames)} in shards)")
    print(f"fps             {info['fps']}")
    print(f"task string     {_task_string(root)!r}")
    print("reference       Diffusion Policy (Chi et al., arXiv 2303.04137), MIT licence")

    print("\n== one sample ==")
    for key in ("observation.state", "action"):
        column = frames[key]
        lo = min(float(v.min()) for v in column)
        hi = max(float(v.max()) for v in column)
        print(f"{key:20s} dim={len(column.iloc[0])}  range=[{lo:.1f}, {hi:.1f}]")
    print("observation.image   96x96x3, AV1 video (decoded on the fly)")

    print("\n== episode shape ==")
    lengths = episodes.size()
    print(
        f"lengths         min={lengths.min()}  p10={int(lengths.quantile(0.1))}  "
        f"median={int(lengths.median())}  p90={int(lengths.quantile(0.9))}  max={lengths.max()}"
    )
    print(f"seconds         {lengths.min() / info['fps']:.1f} - {lengths.max() / info['fps']:.1f} s")

    print("\n== how good are the demonstrations? ==")
    peak_reward = episodes["next.reward"].max()
    # Both gym-pusht and the original LeRobot converter clip reward at one.
    # An unclipped peak can be converted back to coverage. For solved episodes,
    # reward alone supplies a lower bound, so use next.success for the count.
    coverage_lower_bound = peak_reward * SUCCESS_THRESHOLD
    solved = episodes["next.success"].max().astype(bool)
    print(
        f"peak reward     min={peak_reward.min():.3f} median={peak_reward.median():.3f} "
        f"max={peak_reward.max():.3f}"
    )
    print(
        f"peak coverage lower bound  min={coverage_lower_bound.min():.3f} "
        f"median={coverage_lower_bound.median():.3f} max={coverage_lower_bound.max():.3f}"
    )
    print(f"successful episodes       {int(solved.sum())}/{len(solved)} (next.success)")
    print(f"reward saturated episodes {int((peak_reward >= 1).sum())}/{len(peak_reward)}")
    print("The success labels describe recorded states. Evaluate learned policies in fresh simulator episodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
