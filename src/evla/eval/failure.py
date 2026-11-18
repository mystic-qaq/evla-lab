"""Failure taxonomy: not just *whether* an episode failed, but *how*.

A success rate tells you the score. The taxonomy tells you what to fix, and it is
the first thing a hardware team asks after a bad run. Categories are deliberately
coarse, mutually exclusive, and derived only from signals the environment already
reports -- no hand-labelling, so the classification is reproducible for every run.

For Push-T:

``success``
    the environment reported success.
``never_contacted``
    the environment recorded zero contacts (``max_contacts == 0``).
``contact_near_miss``
    a failed episode with recorded contact and peak coverage at least
    ``near_miss_coverage`` (0.85 by default).
``contact_missed``
    a failed episode with recorded contact and peak coverage below the near-miss threshold.

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


NEAR_MISS_COVERAGE = 0.85


def classify_episode(
    episode, near_miss_px: float = 40.0, near_miss_coverage: float = NEAR_MISS_COVERAGE
) -> str:
    """Bucket one episode. Prefers the environment's coverage metric when available."""
    if episode.success:
        return "success"
    info = episode.info or {}
    contacts = info.get("max_contacts")
    coverage = info.get("best_coverage")
    distance = info.get("best_pos_err")
    if contacts is None and coverage is None and distance is None:
        return "failed_unknown"
    if contacts is not None and contacts <= 0:
        return "never_contacted"
    if coverage is not None:
        return "contact_near_miss" if coverage >= near_miss_coverage else "contact_missed"
    if distance is not None and distance <= near_miss_px:
        return "contact_near_miss"
    return "contact_missed"


def taxonomy(
    result: EvalResult, near_miss_px: float = 40.0, near_miss_coverage: float = NEAR_MISS_COVERAGE
) -> Taxonomy:
    counts = dict.fromkeys(CATEGORIES, 0)
    for episode in result.episodes:
        counts[classify_episode(episode, near_miss_px, near_miss_coverage)] += 1
    return Taxonomy(label=result.label, episodes=len(result.episodes), counts=counts)


def to_markdown(taxonomies: list[Taxonomy], title: str | None = "Failure taxonomy") -> str:
    header = (
        (f"# {title}\n\n" if title else "")
        + "| config | episodes | success | near miss | touched, missed | never contacted | unknown |\n"
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
        "\n*Near miss* means a failed episode with recorded contact and peak coverage of at least 0.85. "
        "*Never contacted* means the environment recorded zero contacts during the episode.\n"
    )
    return header + "\n".join(rows) + "\n" + note
