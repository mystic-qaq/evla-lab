"""Failure taxonomy: not just *whether* an episode failed, but *how*.

A success rate tells you the score. The taxonomy tells you what to fix, and it is
the first thing a hardware team asks after a bad run. Categories are deliberately
coarse, mutually exclusive, and derived only from signals the environment already
reports -- no hand-labelling, so the classification is reproducible for every run.

For Push-T:

``success``
    the environment reported success.
``never_contacted``
    the end-effector never touched the block (``max_contacts == 0``) -- typically a
    reaching/perception problem, not a manipulation problem.
``contact_near_miss``
    the block was moved and came within ``near_miss_px`` of the goal translation but
    the pose tolerance was not met -- typically an orientation/precision problem.
``contact_missed``
    the block was touched but never got close -- typically a planning or a
    chunk-staleness problem.

For environments that report nothing but success, everything falls back to
``failed_unknown``, which is honest rather than invented.
"""

from __future__ import annotations

from dataclasses import dataclass

from evla.eval.runner import EvalResult

CATEGORIES = ("success", "contact_near_miss", "contact_missed", "never_contacted", "failed_unknown")


@dataclass
class Taxonomy:
    label: str
    episodes: int
    counts: dict[str, int]

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "episodes": self.episodes,
            "counts": self.counts,
            "fractions": {
                k: round(v / self.episodes, 3) for k, v in self.counts.items() if self.episodes
            },
        }


def classify_episode(episode, near_miss_px: float = 40.0) -> str:
    if episode.success:
        return "success"
    info = episode.info or {}
    contacts = info.get("max_contacts")
    best = info.get("best_pos_err")
    if contacts is None and best is None:
        return "failed_unknown"
    if contacts is not None and contacts <= 0:
        return "never_contacted"
    if best is not None and best <= near_miss_px:
        return "contact_near_miss"
    return "contact_missed"


def taxonomy(result: EvalResult, near_miss_px: float = 40.0) -> Taxonomy:
    counts = dict.fromkeys(CATEGORIES, 0)
    for episode in result.episodes:
        counts[classify_episode(episode, near_miss_px)] += 1
    return Taxonomy(label=result.label, episodes=len(result.episodes), counts=counts)


def to_markdown(taxonomies: list[Taxonomy], title: str = "Failure taxonomy") -> str:
    header = (
        f"# {title}\n\n"
        "| config | episodes | success | near miss | touched, missed | never contacted | unknown |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for t in taxonomies:
        c = t.counts
        n = max(1, t.episodes)
        rows.append(
            f"| {t.label} | {t.episodes} | {c['success']} ({c['success'] / n:.0%}) | "
            f"{c['contact_near_miss']} | {c['contact_missed']} | {c['never_contacted']} | "
            f"{c['failed_unknown']} |"
        )
    note = (
        "\n*Near miss* = block moved and came within the translation tolerance but the full pose "
        "was not reached. *Never contacted* = the end-effector never touched the block, which "
        "points at reaching/perception rather than manipulation.\n"
    )
    return header + "\n".join(rows) + "\n" + note
