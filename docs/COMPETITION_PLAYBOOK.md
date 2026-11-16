# Competition playbook: from this repository to a real tournament

The competition is scored on **task success rate on a real robot**, with the
pipeline "fine-tune a VLA → optimize inference → deploy → compare success rates".
This document is the bridge between what is implemented here (the pipeline, minus
the hardware) and what a team would actually do with four weeks and one arm.

## 1. What already holds

| competition requirement | status in this repo | evidence |
|---|---|---|
| fine-tune a public VLA | ✅ SmolVLA-450M on Push-T, expert-only parameters | `docs/OPTIMIZATION.md`, `results/` |
| measure whether fine-tuning helped | ✅ paired eval with Wilson CI + McNemar | `results/pusht_sweep/LEADERBOARD.md` |
| optimize inference | ✅ precision / flow steps / chunking / server split, with Pareto curve | `evla.optimize`, `docs/OPTIMIZATION.md` |
| deploy to a real arm | 🟡 HAL + runtime + safety + SO-101 backend, **sim-validated only** | `docs/DEPLOYMENT.md` |
| compare success rates fairly | ✅ seeds, CIs, power analysis, failure counters | `docs/EVAL_PROTOCOL.md` |
| know your failure modes | ✅ safety/policy counters + per-episode JSON | `evla.eval.runner` |

The 🟡 is the honest gap, and it is the first item of the plan below.

## 2. The four-week plan

### Week 1 — hardware bring-up and a baseline that is actually measurable

- Calibrate the arm, fix camera pose, record a **10-episode** teleoperation dataset
  of the competition task. Ten episodes is enough to fine-tune, not enough to
  generalise — that is fine, it establishes the pipeline.
- Run the policy at 2 Hz, `execute_steps=5`, safety on, and record every failure.
- Deliverable: `results/robot_week1/` with success rate + failure taxonomy + video.

**Trap to avoid:** tuning the model before the camera and the action scaling are
right. Fix the data collection rig first; a 3° camera tilt is worth more than 10k
training steps.

### Week 2 — data is the lever, not the architecture

- Scale to 50-100 episodes, covering **variation** (object pose, lighting, clutter,
  human disturbance) rather than repetition of the same scene.
- Fine-tune from the week-1 checkpoint; compare 20 episodes paired against the
  week-1 policy with McNemar.
- Deliverable: success-rate curve vs dataset size (the most persuasive plot in any
  imitation-learning report).

### Week 3 — make it fast enough to be safe

- Measure the real control budget: camera → inference → servo, with the actual
  camera (USB exposure and driver latency dominate; the GPU is often not the
  bottleneck).
- Sweep `execute_steps ∈ {1, 5, 10, 25}`, precision bf16, flow steps 4/10, and
  `torch.compile`.
- Choose the operating point on the Pareto front, not the fastest point.
- Deliverable: the latency/robustness table + the chosen configuration with reasons.

### Week 4 — robustness and the final measurement

- Adversarial evaluation: unseen object poses, partial occlusion, a human bumping
  the table. Report success **per condition**, not one aggregate number.
- Freeze the policy, run the official protocol with ≥50 episodes, publish CIs.
- Deliverable: report + video + a failure-mode appendix.

## 3. Where teams actually lose

1. **No measurement discipline.** A 20-episode run with ±20 pts of uncertainty gets
   ranked by noise, and the team cannot tell whether a change helped. Fix: fix the
   seed list, quote intervals, and use paired tests. (Cheapest possible advantage.)
2. **Latency discovered late.** A policy that needs 400 ms per inference cannot run
   a 10 Hz loop. Discover this in week 1 by measuring the loop, not in the final day.
3. **Action-space mismatch.** Degrees vs radians vs normalised, or an offset in the
   gripper channel: the policy "works" in the dataset viewer and slams the table on
   hardware. Fix: dry run at low rate with rate limiting on.
4. **Overfitting the demo scenes.** Success on the training table configuration is
   not success. Fix: hold out configurations, not just episodes.
5. **Chasing a new model instead of the data.** With ≤100 demos, data quality and
   consistency dominate architecture choice.

## 4. What I would build next, in priority order

1. **Real-hardware closed loop** (`SO101Backend` is written; it needs a calibrated
   arm and one day of validation).
2. **Failure taxonomy from video**: classify each failed episode (no contact, wrong
   object, stalls, overshoot) instead of only counting failures. Failure counting
   tells you the score; the taxonomy tells you what to fix.
3. **Temporal aggregation on hardware**: `temporal_ens` costs inference on every
   step; on a real arm with 30 Hz servos, the smoother trajectory may be worth it.
4. **Quantization**: int8 via `torchao` on the VLM backbone. Not implemented here
   because it needs calibration data and careful accuracy checks — a documented
   TODO rather than an unverified claim.
5. **Multi-task suite**: 5-10 language-conditioned tasks, evaluate per task, report
   the mean with a paired design across tasks.

## 5. Answers to the questions this repository invites

**"You only tested in simulation. How is that useful?"**
The pipeline, the failure instrumentation and the measurement protocol are the
transferable parts. What does not transfer is the success-rate number itself, and
`docs/DEPLOYMENT.md` says so explicitly. A team that reaches the robot on day 1 and
has a trustworthy evaluation on day 2 is ahead of a team that spends week 1
re-implementing chunked inference.

**"Why SmolVLA and not OpenVLA / π0?"**
450M parameters fit on a single 24 GB card with batch 32 at 1.3 step/s, which makes
a 20k-step fine-tune a 4-hour job instead of a multi-day one. It also uses flow
matching like π0, so the action-expert reasoning transfers. The adapter is 250
lines; swapping in another LeRobot policy is a day's work.

**"What did you actually learn?"**
That the bottleneck on a real robot is not the model's accuracy but the *staleness
budget*: how long a plan may be executed blind. That is why the sweep measures
`execute_steps` at constant weights — and why on the toy task the same policy drops
from 77% to 40% success when its 8-step plan is executed open-loop against a
faster-moving target.
