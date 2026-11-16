# evla-lab

**An end-to-end Vision-Language-Action pipeline: fine-tune a 450M VLA, optimize its inference, deploy it through a safety-checked runtime, and measure success rates you can defend.**

[![ci](https://github.com/mystic-qaq/evla-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/mystic-qaq/evla-lab/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

![chunking](assets/chunking_ab.gif)

*Same policy, same scene, two execution strategies. Re-planning every step reaches
the moving target; executing an 8-step chunk open-loop commits to a stale plan and
misses. This trade-off — not raw model accuracy — is what decides control frequency
and success rate on a real arm, and it is what this repository measures.*

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

{{RESULTS_TABLE}}

{{FINDINGS}}

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
receding-1       80.0% [62.7%, 90.5%] (24/30)   593.5 Hz
open_loop-8      53.3% [36.1%, 69.8%] (16/30)   799.9 Hz
temporal_ens-8   73.3% [55.6%, 85.8%] (22/30)   566.7 Hz
```

Same weights, same 30 scenes, three execution strategies: **open-loop execution is
50% faster and 27 points worse** once the target moves. That is the whole argument
for measuring the trade-off instead of optimizing one number.

## Reproducing the VLA study

Requires one 24 GB GPU. Wall-clock on an RTX 3090: ~4.3 h fine-tune + ~25 min sweep.

```bash
bash scripts/setup_env.sh                    # torch + lerobot + evla in .venv
bash scripts/download_assets.sh              # SmolVLA-base (873 MB) + Push-T dataset
evla finetune -c configs/finetune_smolvla_pusht.yaml
evla sweep    -c configs/sweep_pusht.yaml    # 8 configs, all on identical seeds
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
