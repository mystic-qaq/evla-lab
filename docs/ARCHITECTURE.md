# Architecture

## The pipeline in one picture

```
                     ┌──────────────────────────────────────────────────────────┐
                     │                    evaluation (evla.eval)                │
                     │  paired seeds · Wilson CI · McNemar · Pareto plots       │
                     └───────────────▲──────────────────────────▲───────────────┘
                                     │                          │
   ┌──────────────┐      ┌───────────┴────────┐      ┌──────────┴────────────┐
   │  data / sim  │      │     fine-tune      │      │   inference optimize  │
   │  evla.envs   │─────▶│  lerobot-train     │─────▶│     evla.optimize     │
   │  toy · PushT │      │  SmolVLA-450M      │      │ precision·flow·compile│
   └──────────────┘      └────────────────────┘      └──────────┬────────────┘
                                                                │
                                                     ┌──────────▼────────────┐
                                                     │      deployment       │
                                                     │      evla.deploy      │
                                                     │ server · runtime ·    │
                                                     │ HAL · safety          │
                                                     └──────────┬────────────┘
                                                                │
                                              Pymunk sim ◀──────┴──────▶ SO-101 arm
```

Arrow semantics: every box talks to the next one through `evla.types.Observation`
and `evla.types.Policy`. That is the whole interface.

## Module map

| module | responsibility |
|---|---|
| `evla.types` | `Observation`, `StepResult`, `Policy` (chunk-predicting ABC) |
| `evla.envs` | `EmbodiedEnv` ABC, `ToyReachEnv` (no deps), `PushTEnv` (Pymunk) |
| `evla.models` | `SmolVLAPolicy` adapter, `FlowChunkPolicy` (from-scratch flow matching) |
| `evla.data` | demo collection, chunk construction, normalisation stats |
| `evla.control` | `ChunkExecutor`: receding / open-loop / temporal-ensembling |
| `evla.optimize` | latency profiler, capability-probing optimization pipeline |
| `evla.deploy` | policy server + client, fixed-rate runtime, HAL (sim, SO-101), safety |
| `evla.eval` | rollout runner, statistics, failure taxonomy, report, paired leaderboard |
| `evla.cli` | one subcommand per stage; every run writes a JSON artefact |

3.7k lines of source, 550 lines of tests, one external interface (`Policy`) that is
about 100 lines long.

## Five decisions worth defending

**1. Wrap LeRobot's trainer instead of writing a training loop.**
SmolVLA fine-tuning needs the dataset pipeline (`LeRobotDataset` with PyAV video
decoding), feature-name validation, normalisation statistics and the flow-matching
loss. All of that exists, is tested upstream and takes ~4 h of GPU time to even
validate. Re-implementing it would be a copy with new bugs. The contribution of
this repository is the *rest* of the loop — inference cost, deployment, and
statistically defensible measurement — which is where most VLA projects stop.
`evla finetune` prints its command and invokes LeRobot 0.4.4 through the
`evla.models.smolvla_train` adapter. The adapter corrects action padding, physical
action dimensions and valid-element reduction while retaining LeRobot's training loop.

**2. Policies expose chunks, not single actions.**
`Policy.predict_chunk` returns `(chunk_size, action_dim)`. LeRobot's own
`select_action` hides a queue, which makes it impossible to compare *execution
strategies* on identical weights. Exposing the chunk moves the decision into
`ChunkExecutor`, where open-loop, receding-horizon and temporal ensembling can be
measured against each other and against their compute cost.

**3. A toy environment that ships with the repo.**
`ToyReachEnv` is ~200 lines of NumPy: a point mass, a drifting target, a 96×96
rendered observation. It exists so that the *entire* pipeline — collect, train,
evaluate, deploy, report — runs in about a minute on a CPU inside CI. Without it,
either CI tests nothing that matters or it downloads a simulation environment, and both are
worse. Raising the target drift turns the toy into a genuine reactivity stress test
(see `docs/OPTIMIZATION.md`).

**4. Latency is measured in the control loop, with CUDA synchronisation.**
`InstrumentedPolicy` synchronises before and after each `predict_chunk` call.
Without that, an async CUDA timer reports kernel-launch time and can under-report
inference by 3–10×. Every headline number in this repository comes from a
synchronised measurement, and the harness is in `evla/optimize/profiler.py` so it
can be audited rather than trusted.

**5. Success rate is reported with an interval, and comparisons are paired.**
`evla eval` never prints a bare percentage: it prints Wilson bounds and the
episode count. `evla leaderboard` compares runs on identical seeds with McNemar's
test. With 20 episodes, one episode is 5 percentage points — a ranking that ignores
this is a coin flip. See `docs/EVAL_PROTOCOL.md`.

## Data flow for one evaluation episode

```
env.reset(seed) ──▶ Observation(state, images, task)
        │
        ▼
ChunkExecutor.act ──▶ [buffer empty?] ──▶ Policy.predict_chunk ──▶ post-process (unnormalise)
        │                                               │
        │                                      (chunk cached, H-1 steps free)
        ▼
action (action_dim,) ──▶ SafetyLayer.filter ──▶ env.step ──▶ StepResult(success, done)
        │
        └──▶ per-step latency, inference count, safety counters ──▶ EvalResult JSON
```

## Testing strategy

| layer | how it is verified |
|---|---|
| statistics | known closed-form values (Wilson 15/20 → [0.53, 0.89]), monotonicity, symmetry |
| chunk executor | scripted policies with known chunks; inference-count assertions per mode |
| safety layer | NaN, clipping, rate limiting, e-stop latch/release, latency hold |
| toy pipeline | collect → train 250 steps → checkpoint round-trip → evaluate → report |
| deployment | full runtime loop against the sim backend; TCP server round trip |
| CLI | `evla quickstart` runs in CI on a clean checkout (~1 min) |
| VLA path | **not** in CI (needs a GPU + 873 MB checkpoint); validated by `results/` |

The last row is the honest gap, and it is stated rather than hidden: the SmolVLA
path is exercised by the committed experiment artefacts in `results/`, not by CI.
