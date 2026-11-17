# evla-lab

**An end-to-end Vision-Language-Action pipeline: fine-tune a 450M VLA, optimize its inference, deploy it through a safety-checked runtime, and measure success rates you can defend.**

[![ci](https://github.com/mystic-qaq/evla-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/mystic-qaq/evla-lab/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

![chunking](assets/chunking_ab.gif)

*Same policy, same scene, two execution strategies. Against a **moving** target,
re-planning every step wins (77% vs 40% success) because an 8-step plan goes stale.
Against the **static** Push-T task the ranking flips (finding 3 below): there is no
motion to track, and re-planning only re-samples noise. Execution strategy is a
measured choice, not a default — which is the point of this repository.*

---

## Why this exists

Most VLA repositories stop at "here is a checkpoint and a training script". A robot
competition is decided by the *other* four questions:

| stage | question | where it is answered here |
|---|---|---|
| 1. fine-tune | does adapting a public VLA to my task actually help? | `evla finetune`, SmolVLA-450M on Push-T |
| 2. inference | how fast is one decision, and what does speed cost in accuracy? | `evla optimize`, Pareto study |
| 3. deployment | what happens when inference is late, or the policy outputs nonsense? | `evla deploy`, runtime + safety + HAL |
| 4. evaluation | is "3 points better" a real difference or noise? | `evla eval`, Wilson CI + paired McNemar |

Every stage writes a JSON artefact, so a claim is a file, not a memory of a terminal.

## Results

SmolVLA-450M fine-tuned on Push-T for 20k steps (single RTX 3090, 4.3 h), evaluated on
40 paired seeds — 20 for the receding configuration, which needs 300 inferences per
episode. `best coverage` is the mean closest pose overlap reached (success = 0.95).

| config | exec / chunk | success (95% CI) | control Hz | p50 infer | best coverage |
|---|---|---|---:|---:|---:|
| zero-shot checkpoint | 50 / 50 | 0% (0–9%) | 137 | 237 ms | 0.33 |
| fine-tuned, fp32 | 50 / 50 | 5% (1–17%) | 130 | 250 ms | 0.89 |
| fine-tuned, bf16 | 50 / 50 | 2% (0–13%) | 127 | 261 ms | 0.95 |
| + `torch.compile` | 50 / 50 | 8% (3–20%) | **229** | **86 ms** | 0.91 |
| + compile + 4 flow steps | 50 / 50 | 10% (4–23%) | **270** | **54 ms** | 0.95 |
| + compile + 4 steps, exec 20 | 20 / 50 | 10% (4–23%) | 185 | 55 ms | 0.95 |
| receding horizon (exec 1) | 1 / 50 | 0% (0–16%) | 18 | 52 ms | 0.79 |
| temporal ensemble (exec 10) | 10 / 50 | 5% (1–17%) | 18 | 52 ms | 0.94 |

![pusht](assets/pusht_rollout.gif)

*A fine-tuned rollout (episode 0 of the fixed seed list, nothing cherry-picked). The
block is pushed onto the target but lands ~45° off, i.e. coverage ≈ 0.9 against the
0.95 success threshold. This is what a "near miss" looks like, and why the tables below
report coverage alongside success.*

Full artefacts: [`results/pusht_sweep/REPORT.md`](results/pusht_sweep/REPORT.md),
[`LEADERBOARD.md`](results/pusht_sweep/LEADERBOARD.md) (paired McNemar),
[`results/pusht_learning_curve/`](results/pusht_learning_curve),
[`results/pusht_lora/`](results/pusht_lora).

**1. Fine-tuning changes the failure mode, not just the score.**
Pose coverage goes 0.33 → 0.95, and the failure taxonomy flips: the pre-trained
checkpoint **never touches the block in 29 of 40 episodes**; the fine-tuned one never
fails to reach it and instead lands within a few degrees of the target in **36 of 40**.
Success goes 0/40 → 4/40 — a 10-point gain that this sample size cannot formally
resolve (paired McNemar p = 0.125; 160 episodes per arm would be needed for 10% vs
2.5%). The coverage and taxonomy numbers carry the claim, not the success rate.

**2. `torch.compile` is the single largest win — up to 4.8x — and it costs nothing measurable.**
237 ms → 261 ms with bf16 (no, bf16 is not faster at batch 1 here) → 86 ms compiled →
54 ms compiled with 4 flow steps. The action expert is a chain of small shape-stable
matmuls, so eager launch overhead dominates. Without this, a 10 Hz loop is impossible;
with it, the loop uses 3% of its budget (see the deployment run below).

**3. Re-planning every step makes this policy *worse*, not better.**
Same weights: open-loop chunk execution reaches coverage 0.95, receding horizon only
0.79. A flow-matching policy re-samples its action noise on every call, so re-planning
20 times a second injects jitter instead of feedback. "Lower latency is always safer"
is false — the right question is how much plan staleness the task tolerates.

**4. More fine-tuning is not monotone — and a warm restart can undo it.** Coverage over
checkpoints: 0.49 (4k) → 0.76 (8k) → 0.92 (12k) → **0.73 (16k)** → 0.95 (20k).
Continuing the same run for another 5k steps at half the learning rate with a fresh
cosine schedule made it *worse*: 0.91 at 22.5k, 0.83 at 25k
([`results/pusht_stage2/`](results/pusht_stage2)). The 20k point was the optimum of that
schedule, not of "more steps". Select checkpoints on validation coverage, never by step
count.

**5. Parameter-efficient tuning is capacity-limited here.** At matched steps, LoRA r=16
(2.4M trainable) plateaus at coverage 0.48 while the default expert-only fine-tune
(≈100M trainable) climbs 0.49 → 0.76. For a domain shift this large, low-rank adapters
need more rank or more data, not just more steps.

## The data: where it comes from, and what it can teach

`lerobot/pusht` is not ours. It is the Push-T demonstration set released with Diffusion
Policy (Chi et al., arXiv 2303.04137, MIT licence), repackaged in the LeRobot format.
The original zarr archive is still at diffusion-policy.cs.columbia.edu; the Hub copy
changes the container (parquet + AV1 video), renames the keys to
`observation.state` / `observation.image` / `action`, and adds one task string. The
demonstrations are human teleoperation; the paper documents a 3Dconnexion SpaceMouse for
its real-robot stations and does not state the device for the simulator set.

`python scripts/audit_dataset.py --root data/pusht` prints all of this from the copy on
disk. Three properties matter more than the size:

- **Episodes are 49-246 frames (median 122), not 300.** That is 5-25 s of human control,
  ending when the operator decided the block was in place. A 300-step evaluation asks the
  policy to stay stable for 2.5x longer than the median demonstration.
- **The action is an absolute end-effector target, not a delta**, and the state is a
  position with no velocity. Velocity has to be inferred from context -- which is why the
  toy environment in this repo carries target velocity explicitly, and why a policy here
  reacts late to a block that is already rolling.
- **No demonstration reaches the success threshold.** The reward column is
  `clip(coverage / 0.95, 0, 1)`; converted back, peak coverage runs 0.77-0.90 (median
  0.85) and **0 of 206 crosses 0.95**. The reading is insensitive to which convention the
  converter used: no reward value reaches 1.0, so no episode ever triggered the clipped
  success value.

That last point reframes the whole result. The policy is fitted to trajectories that
never solve the task themselves, so it learns to stop where the human stopped. Any
success it reaches -- the 4/40 in the sweep -- is generalisation past the demonstrations,
not imitation of them. It also means "improve the demonstrations" is not a hand-wave: the
teacher's own ceiling is 0.90, and no amount of extra fitting moves it.

## Why 10%? A diagnosis, not an excuse

Push-T counts an episode as solved only when the block's pose IoU with the goal exceeds
0.95 -- position *and* orientation. A T-block rotated by ~30 degrees is already below
0.8. The fine-tuned policy sits at 0.95 mean best coverage: it repeatedly lands *on*
the threshold without crossing it on enough frames.

The failure taxonomy says the same thing from the other side. Zero-shot **never touches
the block in 29 of 40 episodes**; the fine-tuned policy never fails to reach it and is
a near miss (coverage >= 0.85) in **36 of 40**. This is a precision failure, not a
competence failure.

Five plausible explanations were tested and rejected:

| hypothesis | measurement | verdict |
|---|---|---|
| the 50-action plan goes stale | execute 50 / 20 / 10 all give 4/40 | not the cause |
| re-planning every step fixes it | receding: 0/20, coverage 0.95 -> 0.79 | worse, not better |
| too few denoising steps | 10 steps 0.95 vs 4 steps 0.91 | inside the noise at n=40 |
| simply undertrained | stage-2 continuation: 0.91 -> 0.83 | worse |
| model capacity | LoRA r16 stalls at 0.48, expert-only reaches 0.95 | mattered, is no longer the limit |

What is left is data -- and unlike the five rows above, it can be measured directly: no
demonstration in the set crosses 0.95 either, and the median one stops at 0.85 (see the
data section above). Four GPU-hours on two hundred demonstrations teach "push the block
onto the target", not "when the pose is 10 degrees off, nudge it this way". The last five
percent of contact-rich precision lives in correction segments the teacher never
performed.

Two honest caveats. 4/40 has a Wilson interval of 4-23%, and the same checkpoint has
scored 2/10 in one run and 0/20 in another -- "10%" is an order of magnitude, not a
measurement. And `torch.compile` changes kernel fusion, which is enough to move a
chaotic contact trajectory; that is why every table here compares configurations on the
same seed list rather than individual episodes.

What would actually move it, ordered by expected payoff: better demonstrations (the only
real lever), checkpoint selection on validation coverage (12k scores 0.92 while 16k
scores 0.73 -- 19 free points), and 160 episodes per arm before claiming any 10-point
difference.

## Deployment run: simulator, 10 Hz, safety layer on

```bash
evla deploy -c configs/deploy.yaml      # fixed-rate loop + policy server + HAL + safety
```

```
steps                 900        (3 episodes x 300, at a real 10 Hz)
inference calls        45        (one per 20 executed actions)
deadline misses         0        (mean budget use 3.1%, worst step 76 ms of 100 ms)
safety                 12 / 900 actions rate-limited, 0 e-stop events
wall time            89.8 s      (== 900 steps / 10 Hz, exactly as scheduled)
```

That is the difference between "the model runs" and "the loop holds its rate": the
budget is spent on inference *amortised over chunks*, and when a policy output jumps
more than 80 px the safety layer rate-limits it instead of letting the arm do it.
Raw report: [`results/deploy_sim.json`](results/deploy_sim.json).

## Quickstart — no GPU, no robot, ~1 minute

```bash
git clone https://github.com/mystic-qaq/evla-lab && cd evla-lab
pip install -e ".[dev]" torch --extra-index-url https://download.pytorch.org/whl/cpu

evla doctor          # what is installed, is CUDA visible
evla quickstart      # collect -> train -> evaluate -> report, all on CPU
```

`evla quickstart` runs the *entire* pipeline on a dependency-free toy task and
writes `outputs/quickstart/eval/REPORT.md`. It is also what CI runs, which is why
the pipeline cannot silently rot.

```bash
evla toy-eval --policy outputs/quickstart/toy_flow.npz --episodes 30 --drift 0.03
```

```
receding-1       76.7% [59.1%, 88.2%] (23/30)   582.4 Hz
open_loop-8      40.0% [24.6%, 57.7%] (12/30)   802.7 Hz
temporal_ens-8   60.0% [42.3%, 75.4%] (18/30)   575.0 Hz
```

Same weights, same 30 scenes, three execution strategies: **open-loop execution is
38% faster and 37 points worse** once the target moves. Temporal ensembling recovers
most of the accuracy at the same control rate as re-planning. That is the whole
argument for measuring a trade-off instead of optimizing one number.

## Reproducing the VLA study

Requires one 24 GB GPU. Wall-clock on an RTX 3090: 4.3 h fine-tune, ~10 min learning
curve, ~35 min optimisation matrix, ~2 min deployment run.

```bash
bash scripts/setup_env.sh                    # torch + lerobot + evla in .venv
bash scripts/download_assets.sh              # SmolVLA-base (873 MB) + Push-T dataset
evla finetune -c configs/finetune_smolvla_pusht.yaml
evla sweep    -c configs/sweep_learning_curve.yaml   # 6 checkpoints, 20 seeds each
evla sweep    -c configs/sweep_pusht.yaml            # 8 configs, 40 seeds each
evla deploy   -c configs/deploy.yaml                 # fixed-rate loop, 10 Hz
```

`evla sweep` produces `results/pusht_sweep/{REPORT.md, LEADERBOARD.md, *.json, *.png}`.

## One policy, four interfaces

```
                    Observation (state + images + task)
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │                      Policy                         │
        │        predict_chunk(obs) -> (H, action_dim)        │
        └───┬───────────────┬──────────────────┬──────────────┘
            │               │                  │
     ┌──────▼─────┐  ┌──────▼──────┐   ┌───────▼────────┐
     │ChunkExecutor│  │ Instrumented│   │  RemotePolicy  │
     │ receding /  │  │  (latency)  │   │  (TCP server)  │
     │ open-loop / │  └─────────────┘   └───────┬────────┘
     │ temporal-ens│                            │
     └──────┬──────┘                     ┌──────▼───────┐
            │                            │ PolicyServer │
     ┌──────▼──────┐                     └──────────────┘
     │ SafetyLayer │  clamp · rate-limit · watchdog · e-stop
     └──────┬──────┘
     ┌──────▼──────┐
     │  HAL        │  MuJoCo sim  ·  SO-101 arm
     └─────────────┘
```

Because everything speaks `Observation`/`Policy`, a 450M SmolVLA, a 2M flow-matching
MLP and a scripted controller are interchangeable — including inside unit tests.

## Repository layout

```
src/evla/
  types.py            Observation, StepResult, Policy            (the whole interface)
  envs/               toy_reach (numpy, no deps) · pusht (MuJoCo)
  models/             smolvla adapter (450M) · flow_chunk (from scratch, 2M)
  data/               demo collection, chunk construction, normalisation
  control/            ChunkExecutor: receding / open_loop / temporal_ens
  optimize/           latency profiler · capability-probing optimization pipeline
  deploy/             runtime · policy server/client · HAL (sim, SO-101) · safety
  eval/               rollout runner · statistics · report · paired leaderboard
  cli.py              evla doctor/quickstart/finetune/bench/eval/sweep/serve/deploy
configs/              every experiment is a YAML file, not a shell history
docs/                 architecture · optimization study · deployment · eval protocol · playbook
results/              committed experiment artefacts (JSON + figures + reports)
tests/                39 tests, CPU only, ~40 s
```

## Design decisions in one line each

- **Wrap LeRobot's trainer, own the rest.** Re-implementing SmolVLA fine-tuning
  would be a copy with new bugs; inference cost, deployment and measurement are
  where the value is. `evla finetune` prints the exact upstream command it runs.
- **Policies return chunks, not actions.** Execution strategy becomes an explicit,
  measurable choice instead of a hidden queue inside the model wrapper.
- **Latency is measured with `cuda.synchronize()` inside the control loop**, because
  an async timer under-reports inference by 3-10x and p95 — not the mean — decides
  whether the robot stalls.
- **Clipping is counted, not fatal.** A badly calibrated policy is a calibration
  problem; the e-stop is reserved for non-finite actions.
- **No bare percentages.** Every success rate ships with a Wilson interval and an
  episode count; comparisons use identical seeds and McNemar's exact test.

More: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md),
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md),
[`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md),
[`docs/COMPETITION_PLAYBOOK.md`](docs/COMPETITION_PLAYBOOK.md).

## What this does *not* claim

- **No real robot was used.** Every number is simulated. The HAL, runtime, safety
  layer and SO-101 backend are written and sim-validated; hardware validation is the
  first item of the roadmap.
- **Push-T is a small benchmark.** It is the right size for studying inference cost
  and execution strategy in 12 hours; it is not evidence of manipulation skill.
- **The absolute success rate is not the point.** The pipeline, the failure
  instrumentation and the measurement protocol are.

## Roadmap

1. SO-101 hardware bring-up: calibration, action-scaling, 2 Hz dry runs, failure taxonomy.
2. Data scaling study: success rate vs number of demonstrations (the most persuasive
   plot in imitation learning).
3. int8 quantization of the VLM backbone with an accuracy check (needs calibration data).
4. Multi-task suite with per-task CIs instead of one aggregate number.

## References

- SmolVLA (Shukor et al., 2025) — the 450M VLA this repo fine-tunes.
- ACT / temporal ensembling (Zhao et al., 2023) — the chunk-averaging baseline.
- Flow Matching (Lipman et al., 2023), π0 (Black et al., 2024) — action-expert formulation.
- Diffusion Policy (Chi et al., 2023) — action chunking as a variance-reduction device.
- LeRobot — dataset, training and hardware SDK.

MIT licensed.
