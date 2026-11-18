"""Merge human and successful simulated Push-T demonstrations with explicit sampling weights."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def verify(root: Path, episodes: int, frames: int) -> None:
    dataset = LeRobotDataset("local/pusht-mixed-balanced", root=root, video_backend="pyav")
    if dataset.num_episodes != episodes or len(dataset) != frames:
        raise ValueError(f"incomplete or different mixture at {root}")
    dataset[0]
    dataset[len(dataset) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-root", type=Path, default=Path("data/pusht"))
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--human-repeat", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.human_repeat < 1:
        raise ValueError("human-repeat must be positive")
    output = args.out.resolve()
    for source in (args.human_root, args.teacher_root):
        if output.is_relative_to(source.resolve()):
            raise ValueError("the output must be outside the source datasets")

    human_info = json.loads((args.human_root / "meta/info.json").read_text(encoding="utf-8"))
    teacher = LeRobotDataset("local/pusht-diffusion-success-large", root=args.teacher_root, video_backend="pyav")
    if human_info["fps"] != teacher.fps:
        raise ValueError("source datasets have different frame rates")
    if human_info["features"].keys() != teacher.features.keys():
        raise ValueError("source datasets have different features")
    for key, feature in human_info["features"].items():
        target = teacher.features[key]
        if (feature["dtype"], tuple(feature["shape"])) != (target["dtype"], tuple(target["shape"])):
            raise ValueError(f"source datasets disagree on {key}")

    episodes = args.human_repeat * human_info["total_episodes"] + teacher.num_episodes
    frames = args.human_repeat * human_info["total_frames"] + len(teacher)
    if output.exists():
        verify(output, episodes, frames)
        print(f"verified existing mixture with {episodes} episodes and {frames} frames")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pusht-mixture-", dir=output.parent) as temporary:
        staging = Path(temporary)
        compatible_human = staging / "human"
        shutil.copytree(args.human_root, compatible_human, ignore=shutil.ignore_patterns(".cache"))
        # v3 conversions use different metadata spelling for the same AV1 video features.
        human_info["features"] = teacher.features
        human_info["robot_type"] = teacher.meta.robot_type
        (compatible_human / "meta/info.json").write_text(json.dumps(human_info, indent=2), encoding="utf-8")
        human = LeRobotDataset("lerobot/pusht", root=compatible_human, video_backend="pyav")
        merged_root = staging / "merged"
        merge_datasets(
            [human] * args.human_repeat + [teacher], "local/pusht-mixed-balanced", output_dir=merged_root,
        )
        verify(merged_root, episodes, frames)
        (merged_root / "provenance.json").write_text(json.dumps({
            "human_root": str(args.human_root.resolve()),
            "teacher_root": str(args.teacher_root.resolve()),
            "human_repeat": args.human_repeat,
            "unique_episodes": human.num_episodes + teacher.num_episodes,
            "sampled_episodes": episodes,
            "sampled_frames": frames,
        }, indent=2), encoding="utf-8")
        try:
            merged_root.rename(output)
        except OSError:
            if not output.exists():
                raise
            verify(output, episodes, frames)
    verify(output, episodes, frames)
    print(f"prepared {episodes} sampled episodes and {frames} frames")


if __name__ == "__main__":
    main()
