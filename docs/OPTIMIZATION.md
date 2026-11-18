# Inference optimization study: what actually moves the control frequency

The question is not "how fast is the model". It is:

> **At a fixed success rate, how many decisions per second can this policy sustain —
> and which knob buys the most speed per point of accuracy?**

This document is the experimental design, the measurement protocol, and the results.

The historical Push-T peak coverage and derived failure categories in this document
predate the episode-reset fix. Corrected values are in the README and retraining results.

## 1. The four knobs under test

| knob | what it changes | expected cost | where |
|---|---|---|---|
| `precision` fp32 → bf16 | matmul dtype, no algorithmic change | ~0 (rounding only) | `SmolVLAPolicy.set_precision` |
| `num_flow_steps` 10 → 4 → 2 | Euler steps in the flow-matching action expert | plan quality, more at 2 | `SmolVLAPolicy.set_num_flow_steps` |
| `execute_steps` 1 → 20 → 50 | how much of a predicted chunk is executed before re-planning | reactivity (staler plan) | `ChunkExecutor` |
| `mode` receding / open-loop / temporal-ens | how overlapping chunks are combined | smoothness vs. inference count | `ChunkExecutor` |
| `compile` | `torch.compile` on the action-chunk denoise loop | none measured, huge win | `SmolVLAPolicy.enable_compile` |

Deliberately **not** in scope: quantization (needs calibration data and an accuracy
check — an honest TODO, not a claim), distillation, and TensorRT.

## 2. Measurement protocol

Anything that measures "latency" without stating these four things is not a
measurement:

1. **Synchronised timing.** `torch.cuda.synchronize()` before and after each
   `predict_chunk`. Async timing measures kernel launches, not inference.
2. **Warmup discarded.** 3 calls before the first recorded sample (cuDNN autotune,
   lazy init, one-time allocations).
3. **Same observation.** A fixed frame from a seeded environment, so the input is
   not a variable.
4. **Percentiles reported.** p50/p95/p99 plus the implied control frequency
   `1000 / p95`.

Two levels of measurement, and they answer different questions:

* `evla bench` — pure model latency (microbenchmark, one config at a time).
* `evla eval` — latency **inside the loop**, which additionally contains observation
  conversion, normalisation, environment stepping and memory transfers. The
  difference between the two is the part that surprises people.

## 3. Why chunking dominates everything else

At 10 Hz a control step has 100 ms. A single SmolVLA forward pass on an RTX 3090 at
batch 1 costs, measured with CUDA synchronisation:

| configuration | p50 per forward | implied rate | source |
|---|---:|---:|---|
| fp32, 10 flow steps | ~340 ms | 3 Hz | `ft__fp32__ope50` |
| bf16, 10 flow steps | ~255 ms | 4 Hz | `ft__bf16__ope50` |
| bf16, 4 flow steps | ~135 ms | 7 Hz | `ft__bf16__flow4__ope50` |
| bf16, 4 flow steps, compiled | ~50 ms | 20 Hz | `ft__bf16__compile__flow4__ope50` |

`torch.compile` is the single largest win -- nearly **2.7x** over eager bf16 at the same
weights -- because the flow-matching head is a long chain of small, shape-stable
matmuls that eager mode launches one kernel at a time. That result is what makes a
10 Hz+ control loop possible on one consumer GPU, and it is why the sweep includes a
compiled configuration rather than treating compile as an afterthought.

Chunking then decides how that budget is spent:

| execution | inferences per 300-step episode | effective control rate | plan staleness |
|---|---:|---|---|
| receding (execute 1) | 300 | ~20 Hz at p50, cannot hold under load | 0 steps |
| open loop (execute 20) | 15 | 10 Hz + generous margin | up to 2 s |
| open loop (execute 50) | 6 | 10 Hz comfortable | up to 5 s |

The model is not the bottleneck — **the amortisation is**. That is why the sweep
varies `execute_steps` at fixed weights: it is the cheapest speedup available and
the one with a real accuracy cost.

## 4. Results

Measured on one RTX 3090, Push-T, SmolVLA-450M fine-tuned 20k steps, 40 paired seeds
(20 for receding). Full artefacts: `results/pusht_sweep/{REPORT.md, LEADERBOARD.md}`.

| config | exec / chunk | success (95% CI) | control Hz | p50 | p95 | steps per inference |
|---|---|---|---:|---:|---:|---:|
| zero-shot | 50 / 50 | 0% (0–9%) | 137 | 237 ms | 250 ms | 50 |
| ft, fp32 | 50 / 50 | 5% (1–17%) | 130 | 250 ms | 265 ms | 50 |
| ft, bf16 | 50 / 50 | 2% (0–13%) | 127 | 261 ms | 265 ms | 50 |
| ft, bf16, compile | 50 / 50 | 8% (3–20%) | 229 | 86 ms | 90 ms | 50 |
| ft, bf16, compile, flow 4 | 50 / 50 | 10% (4–23%) | 270 | 54 ms | 54 ms | 50 |
| ft, bf16, compile, flow 4 | 20 / 50 | 10% (4–23%) | 185 | 55 ms | 56 ms | 20 |
| ft, bf16, compile, flow 4, receding | 1 / 50 | 0% (0–16%) | 18 | 52 ms | 53 ms | 1 |
| ft, bf16, compile, flow 4, temporal ens | 10 / 50 | 5% (1–17%) | 18 | 52 ms | 54 ms | 1 |

The historical coverage measurements contain the episode-reset bug. Corrected
20-seed learning-curve peak means are 0.507 at 12k, 0.390 at 16k, 0.436 at 20k,
and 0.460 at 25k. These measurements support checkpoint selection using validation.

Repeating the original 40 seeds with the 20k checkpoint gives 4/40 successes for
both execution lengths. Mean peak coverage is 0.647 with execute 20 and 0.492 with
execute 50. With execute 20, 16 of the 36 failures reach 0.85 coverage.

## 5. Reading the results

- Historical median latency is 261 ms for eager bf16, 86 ms with compilation,
  and 54 ms with compilation and four flow steps. Each success-rate estimate has
  its own uncertainty and requires a paired comparison.
- The historical 40-seed sweep has four successes for the best configuration and
  zero for the zero-shot reference. Its paired exact McNemar p-value is 0.125.
- Corrected peak coverage varies with execution length. The new retraining run
  selected a 16-step prediction with eight executed actions and four flow steps.
  On 200 held-out seeds it succeeds 65 times, compared with 16 for the old model.

Full candidate results, confidence intervals, and the frozen final test are in the
[retraining report](../results/pusht_retrain_20261118/TRAINING_REPORT.md).

## 6. What the numbers imply for a real deployment

At 10 Hz, executing 50 actions consumes five seconds of simulated time before the
next observation-dependent plan. Choose this interval using task validation.
For the retrained model, the measured p95 inference latency is 53.61 ms and the
maximum is 515.35 ms. Deployment timing must account for the measured tail.
The evaluation's `control_hz` measures offline simulator throughput. The physical
simulation step remains 0.1 seconds.

## 7. Reproducing

```bash
evla bench -c configs/sweep_pusht.yaml            # microbenchmark, one config
evla sweep -c configs/sweep_pusht.yaml            # the full matrix + report
evla report --results results/pusht_sweep         # regenerate figures without a GPU
```
