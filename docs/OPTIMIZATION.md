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
| `compile` | `torch.compile` on the denoise loop | unverified in this model | `CompiledPolicy` |

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

At 10 Hz a control step has 100 ms. One SmolVLA forward pass takes ~100-140 ms on an
RTX 3090 at batch 1. So:

| execution | inferences per 300-step episode | effective control rate | plan staleness |
|---|---:|---|---|
| receding (execute 1) | 300 | ≤ 7 Hz (cannot hold 10 Hz) | 0 steps |
| open loop (execute 20) | 15 | 10 Hz + margin | up to 2 s |
| open loop (execute 50) | 6 | 10 Hz comfortable | up to 5 s |

The model is not the bottleneck — **the amortisation is**. That is why the sweep
varies `execute_steps` at fixed weights: it is the cheapest speedup available and
the one with a real accuracy cost.

## 4. Results

<!-- RESULTS -->

## 5. Reading the results

Three claims survive the noise; the rest are inside the confidence intervals and are
reported as "no detectable difference" rather than as wins.

<!-- FINDINGS -->

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
