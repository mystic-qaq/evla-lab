# Deployment: from `predict_chunk` to an arm that does not break

## 1. The loop

`evla.deploy.runtime.DeployRuntime` is a fixed-rate loop:

```
for each episode:
    backend.reset_episode(seed)
    for each control step (period = 1/control_hz):
        t0 = now
        action = executor.act(backend.read())        # policy inference only when the chunk is empty
        action = safety.filter(action, latency)      # clamp / rate-limit / watchdog
        backend.send(action)                         # actuators (or env.step in sim)
        budget_used += (now - t0) / period
        if (now - t0) > period: deadline_misses += 1
        sleep until the deadline                     # fixed rate, not "sleep(0.1)"
```

Two details that matter more than they look:

* **Deadline scheduling.** `sleep(period)` after doing work accumulates drift and
  silently lowers the control rate. The loop sleeps until an absolute deadline and
  counts misses, so the reported frequency is measured, not assumed.
* **Chunk amortisation is inside the budget.** 300 steps at 10 Hz with
  `execute_steps=50` needs 6 inferences, not 300. The report states
  `inference_calls_per_step` so the amortisation is visible.

## 2. The timing budget, honestly

Measured on one RTX 3090 (bf16, 4 flow steps, 96×96 input, batch 1):

| quantity | value | consequence |
|---|---|---|
| one `predict_chunk` (H=50) | ~100–140 ms | fits a 10 Hz period (100 ms) only marginally |
| env step (MuJoCo Push-T) | ~1–5 ms | negligible |
| TCP round trip, localhost | ~0.3 ms | negligible; not the bottleneck |
| chunk horizon in *seconds* | 50 / 10 Hz = **5 s** | the arm is blind for up to 5 s if executed open-loop |

That last row is the real deployment problem. It is why `execute_steps` exists and
why the sweep measures `execute_steps ∈ {1, 20, 50}` at identical weights: **the
correct question is not "what is the inference latency" but "how stale may the
plan get before success drops".**

## 3. Safety layer

Order of defence (`evla/deploy/safety.py`, fully unit-tested):

1. **E-stop (latched).** Once engaged, no policy output reaches the actuators.
2. **Finiteness.** NaN/Inf replaced by the last safe action; repeated → e-stop.
3. **Position limits.** Workspace or joint bounds, per dimension.
4. **Rate limiting.** `|a_t − a_{t−1}| ≤ max_delta`, which converts a hallucinated
   40 cm jump into a bounded motion.
5. **Watchdog.** If inference exceeds `max_latency_ms`, hold the last action.

Design note: clipping and rate limiting are **counted, not fatal**. A policy whose
outputs need clamping is badly calibrated, not dangerous — escalating to an e-stop
on the fifth clip turns a bad policy into a dead robot, which hides the actual
problem. The e-stop is reserved for non-finite actions and (optionally, via
`estop_on_repeated_clip`) for repeated clipping if a team wants the stricter
behaviour.

## 4. Policy server

`evla serve` runs the policy in its own process; `evla deploy --remote` connects to
it. Why:

* the control loop never blocks on a 450M forward pass — it decides what to do when
  inference is late (hold position) instead of freezing;
* one GPU host can serve several arms / evaluation workers;
* it creates a measurable seam: round trip = queue + inference + serialisation.

Transport is `multiprocessing.connection` over TCP with an HMAC handshake
(stdlib, length-prefixed pickle). Appropriate for a trusted lab network; swap the
transport for gRPC or ZMQ outside one. **Pickle executes code — never expose the
port to an untrusted network.**

Inference is serialised on a lock: GPU inference is not re-entrant, and pretending
otherwise produces OOM errors under load.

## 5. Bring-up checklist for a real SO-101

The `SO101Backend` implements the same three-method HAL as the simulator, so the
runtime code is identical. What changes is everything around it:

- [ ] **Calibrate** the follower (`lerobot-calibrate`) and record the arm's
      motor order — the HAL maps actions to `.pos` fields by name, and a
      re-ordered list silently rotates the action vector.
- [ ] **Units.** SO-101 reports degrees. Decide once where normalisation happens
      (`action_scale`, `action_offset` in the backend) and write it down. A factor
      of 100 here is the difference between a gentle push and a smashed gripper.
- [ ] **Workspace limits.** Fill `safety.action_min/max` from the calibrated range,
      not from the training dataset.
- [ ] **`max_relative_target`.** Keep LeRobot's vendor-level rate limit on as a
      second line of defence (~5° per command is a sane start).
- [ ] **Camera pose.** Match the training dataset's camera height/angle. A policy is
      far more sensitive to viewpoint than to lighting.
- [ ] **Task string.** SmolVLA is language-conditioned; a different instruction
      string is a different task to the model.
- [ ] **Dry run at 2 Hz** with `realtime: true` and one episode before letting the
      policy drive at the target rate.
- [ ] **Log everything**: `RuntimeReport` JSON per run, video, safety counters.

## 6. Failure modes this harness is designed to expose

| symptom | where it shows up | likely cause |
|---|---|---|
| deadline miss rate ≈ 100% | `RuntimeReport.miss_rate` | receding-horizon execution on a slow GPU |
| high `rate_limited` count | `safety.rate_limited` | action-space mismatch (units/scaling) |
| high `clipped` count | `safety.clipped` | workspace limits narrower than the policy's range |
| success fine in sim, zero on hardware | paired eval logs | camera pose, latency, or unmodelled dynamics |
| e-stop events > 0 | `safety.estop_events` | non-finite output (usually fp16 overflow) |

## 7. Sim-to-real gap: what this repository does *not* claim

Everything measured here is in simulation. The honest statement:

* **Measured:** fine-tuning gains, inference cost, chunking trade-off, runtime and
  safety behaviour — all of which transfer qualitatively to hardware.
* **Not measured:** contact dynamics, camera exposure, servo latency, and the
  sim-to-real drop. A Push-T policy evaluated in MuJoCo says nothing about success
  rate on a real arm; it says something about the *pipeline* that will be used.

The next section of the roadmap (`README.md`) is exactly this gap, and closing it is
what the first two weeks of a competition would be spent on.
