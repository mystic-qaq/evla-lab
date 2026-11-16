# Evaluation protocol: measuring success rate you can defend

The competition ranks teams by task success rate. That makes the *measurement*, not
the model, the thing that decides the ranking — so this document is the protocol,
and `evla.eval` is its implementation.

## 1. What counts as a success

Success is decided **by the environment adapter**, never inferred afterwards:

| env | success source |
|---|---|
| `pusht` | `info["is_success"]` from `gym_pusht` (block within pose tolerance of the goal) |
| `toy_reach` | distance to the drifting target < `success_radius` (checked every step) |

Two rules that prevent the classic silent inflation of success rates:

* **Latched, not instantaneous.** In this repo a success is recorded once the
  environment reports it; a policy that touches the goal at step 40 and drifts away
  by step 300 is still a success. That is the standard convention — but it must be
  *stated*, because "final-state grading" and "any-time grading" differ by tens of
  points on real tasks.
* **`max_steps` is part of the task definition.** Every experiment fixes it (300 for
  Push-T, 48 for the toy). Changing the horizon changes the success rate, so it is
  recorded in the result JSON.

## 2. How many episodes

A success rate from `n` episodes is a binomial estimate. Its 95% Wilson interval at
50% success:

| episodes | 95% CI | half-width |
|---:|---|---:|
| 10 | [23.7%, 76.3%] | ±26 pts |
| 20 | [29.9%, 70.1%] | ±20 pts |
| 50 | [36.6%, 63.4%] | ±13 pts |
| 100 | [40.4%, 59.6%] | ±10 pts |

Even 100 episodes cannot resolve a 10-point gap cleanly. Run the calculation before
the experiment, not after:

```
$ evla ... # or in Python
>>> from evla.eval.stats import required_episodes
>>> required_episodes(0.60, 0.70)   # 60% vs 70%, alpha=0.05, power=0.8
354                                  # per arm
>>> required_episodes(0.60, 0.65)   # a 5-point gap
1468
>>> required_episodes(0.30, 0.80)   # a coarse gap is cheap to detect
12
```

Practical consequence for a 12-hour experiment budget: **coarse comparisons
(does fine-tuning work at all?) need ~20 episodes; fine comparisons (bf16 vs fp32)
need hundreds.** So this repository uses:

* 20 episodes for the headline comparisons (`docs/OPTIMIZATION.md`),
* explicitly *not* claiming significance for differences smaller than the CI, and
* McNemar with identical seeds, which is far more sensitive than an unpaired test
  because it ignores scene-to-scene variance.

## 3. Paired seeds

Every configuration is evaluated on the **same** seed list (`seed0 .. seed0+n-1`).
This is what allows a paired test:

```
seeds:        0  1  2  3  4  5  6  7
A (fp32):     1  0  1  1  0  0  1  1     -> 5/8
B (bf16):     1  1  1  1  0  1  1  1     -> 7/8
discordant:      +1 +0 +0 +0 +0 +1 +0 +0  -> b=0, c=2, p(McNemar)=0.5
```

An unpaired z-test on 5/8 vs 7/8 would return p≈0.32 with almost no power. The
paired view says: *the two policies disagree on exactly two scenes, both in the same
direction* — which is the actual evidence. `evla leaderboard` reports
`wins/losses` and the exact McNemar p-value for exactly this reason.

## 4. What is measured besides success

| metric | why it matters | where |
|---|---|---|
| `control_hz` | effective closed-loop rate = 1 / mean step time | `EvalResult.control_hz` |
| `p50/p95/p99` inference | p95 decides whether the arm stalls | `InstrumentedPolicy.stats` |
| `action_reuse` | steps executed per inference (chunk amortisation) | `ChunkExecutor.stats` |
| deadline miss rate | fraction of cycles where inference exceeded the period | `RuntimeReport.miss_rate` |
| safety counters | clipping / rate-limits / e-stops during the run | `RuntimeReport.safety` |
| `mean_episode_s` | wall-clock cost of an evaluation batch | `EvalResult` |

A configuration is only "better" if it improves success **at the same or lower
latency**. That is the Pareto view in `docs/OPTIMIZATION.md`.

## 5. Common ways these numbers get inflated (and how they are blocked here)

| pitfall | mitigation in this repo |
|---|---|
| reporting the best of N seeds | the seed list is fixed before the run and stored in the JSON |
| tuning on the eval set | the fine-tune sees the training split only; `seed0` sweeps are separate from training |
| timing without `cuda.synchronize()` | `InstrumentedPolicy` always synchronises |
| comparing policies with different horizons | `max_steps` recorded per result and asserted equal |
| "it worked in the video" | videos are recorded from episode 0 of a fixed seed, not cherry-picked |
| silent action clipping making a policy look safe | safety counters are reported next to success |

## 6. Reproducing a claim

Every table row in this repository maps to a JSON file:

```
results/pusht_sweep/ft__bf16__ope50.json     # config, per-episode records, latency, safety
results/pusht_sweep/REPORT.md                # tables + figures
results/pusht_sweep/LEADERBOARD.md           # paired comparison vs reference config
```

Regenerate the markdown without touching a GPU:

```
evla report --results results/pusht_sweep
```
