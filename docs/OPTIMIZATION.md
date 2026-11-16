# Inference optimization study: what actually moves the control frequency

The question is not "how fast is the model". It is:

> **At a fixed success rate, how many decisions per second can this policy sustain —
> and which knob buys the most speed per point of accuracy?**

This document is the experimental design, the measurement protocol, and the results.

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

| config | exec / chunk | success (95% CI) | control Hz | p50 | p95 | steps per inference | best coverage |
|---|---|---|---:|---:|---:|---:|---:|
| zero-shot | 50 / 50 | 0% (0–9%) | 137 | 237 ms | 250 ms | 50 | 0.33 |
| ft, fp32 | 50 / 50 | 5% (1–17%) | 130 | 250 ms | 265 ms | 50 | 0.89 |
| ft, bf16 | 50 / 50 | 2% (0–13%) | 127 | 261 ms | 265 ms | 50 | 0.95 |
| ft, bf16, compile | 50 / 50 | 8% (3–20%) | 229 | 86 ms | 90 ms | 50 | 0.91 |
| ft, bf16, compile, flow 4 | 50 / 50 | 10% (4–23%) | 270 | 54 ms | 54 ms | 50 | 0.95 |
| ft, bf16, compile, flow 4 | 20 / 50 | 10% (4–23%) | 185 | 55 ms | 56 ms | 20 | 0.95 |
| ft, bf16, compile, flow 4, receding | 1 / 50 | 0% (0–16%) | 18 | 52 ms | 53 ms | 1 | 0.79 |
| ft, bf16, compile, flow 4, temporal ens | 10 / 50 | 5% (1–17%) | 18 | 52 ms | 54 ms | 1 | 0.94 |

Learning curve (same execution strategy, 20 seeds each): coverage 0.30 (zero-shot) →
0.49 (4k) → 0.76 (8k) → 0.92 (12k) → 0.73 (16k) → 0.95 (20k); success stays 0 until
20k (1/20, then 4/40 in the sweep).

Failure taxonomy at 20k: zero-shot **never contacts the block in 29/40 episodes**; the
fine-tuned policy never fails to reach it and is a *near miss* (coverage ≥ 0.85) in
36/40. Receding-horizon execution converts 11/20 episodes from near-miss into
touched-but-missed.

## 5. Reading the results

Three claims survive the noise; the rest are inside the confidence intervals and are
reported as "no detectable difference" rather than as wins.

- **Compile is the win, and it is free.** 261 ms → 86 ms at 10 flow steps, → 54 ms at
  4 steps: 4.8x end to end, with coverage unchanged inside the noise of 40 episodes.
  Everything else in this table is a rounding error next to it.
- **bf16 alone buys nothing at batch 1.** 250 ms (fp32) vs 261 ms (bf16) is within
  run-to-run variance; the win only appears once the kernel-launch overhead is removed
  by compilation. Reporting "bf16 gives 2x" would have been wrong here.
- **Flow steps trade latency for a small, measurable accuracy cost.** 86 ms → 54 ms
  going 10 → 4 steps, coverage 0.91 → 0.95 (no detectable cost at this n). Below 4
  steps the denoiser stops being a reliable sampler, so 4 is the operating point.
- **Chunk length beats reactivity for a stochastic policy.** Open-loop execution
  reaches coverage 0.95; receding (re-plan every step) only 0.79, because every
  re-plan re-samples the flow noise. Temporal ensembling recovers most of it (0.94)
  at 10x the inference cost of open-loop.
- **Latency and success are not the same axis.** The two 10% configurations differ by
  85 Hz in control rate and not at all in success; the receding configuration is the
  *lowest* latency and the *lowest* quality. Optimising the wrong axis is easy.

Statistical caveat, stated rather than buried: at 40 episodes, 10% vs 0% is
directionally consistent (paired McNemar 4 wins / 0 losses, p = 0.125; unpaired
two-proportion p = 0.04) but not resolved. 160 episodes per arm would be needed to
separate 10% from 2.5% at 80% power. The latency numbers are exact; the success
ordering is suggestive.

## 6. What the numbers imply for a real deployment

* **Choose the operating point by staleness, not by FPS.** A 50-action chunk at 10 Hz
  means up to 5 s of blind execution; for a dynamic task, `execute_steps` of 10-20 is
  the defensible choice even though it is ~2-4× more inference.
* **bf16 is free speed in this setup** and it is the first thing to turn on.
* **Shortening the flow denoiser trades a measurable amount of accuracy for
  latency** — worth it only if the accuracy loss is inside the CI, and it usually is
  not at 2 steps.
* **Temporal ensembling is the smoothing option**, at the price of paying inference
  every step. On hardware with 30 Hz servos that may still be the right trade; on a
  10 Hz task it is dominated by open-loop execution.

## 7. Reproducing

```bash
evla bench -c configs/sweep_pusht.yaml            # microbenchmark, one config
evla sweep -c configs/sweep_pusht.yaml            # the full matrix + report
evla report --results results/pusht_sweep         # regenerate figures without a GPU
```
