# Evaluation report

> Historical report generated before the Push-T episode-reset fix. Peak coverage and derived failure categories require re-evaluation. Corrected measurements are in the project README.

| # | config | mode (exec/chunk) | success | 95% Wilson CI | control Hz | p50 lat (ms) | p95 lat (ms) | steps/inference | mean steps | best coverage |
|---|--------|-------------------|---------|---------------|------------|--------------|--------------|-----------------|------------|-----------------|
| 1 | base__zeroshot | open_loop (50/50) | **0%** | 0%-16% | 137.2 | 237.28 | 245.22 | 50.0 | 300 | 0.30 |
| 2 | expertonly__12k | open_loop (50/50) | **0%** | 0%-16% | 127.3 | 264.16 | 270.87 | 50.0 | 300 | 0.92 |
| 3 | expertonly__16k | open_loop (50/50) | **0%** | 0%-16% | 129.1 | 257.28 | 259.56 | 50.0 | 300 | 0.73 |
| 4 | expertonly__20k | open_loop (50/50) | **5%** | 1%-24% | 129.2 | 255.67 | 265.07 | 49.808 | 299 | 0.95 |
| 5 | expertonly__4k | open_loop (50/50) | **0%** | 0%-16% | 129.8 | 255.96 | 259.65 | 50.0 | 300 | 0.49 |
| 6 | expertonly__8k | open_loop (50/50) | **0%** | 0%-16% | 129.4 | 256.22 | 265.21 | 50.0 | 300 | 0.76 |

`control Hz` is the effective closed-loop rate (1 / mean environment step time, including reused actions); `p50/p95 lat` are per-inference latencies measured with CUDA synchronisation; `steps/inference` is how many control steps one forward pass buys. `best coverage` is the mean best pose overlap reached during an episode (success needs 0.95) -- a graded signal that keeps moving when binary success is still stuck at zero.

## Figures

### success

![success](success_rates.png)

### coverage

![coverage](coverage.png)

### pareto

![pareto](pareto.png)

### latency

![latency](latency.png)

### learning_curve

![learning_curve](learning_curve.png)

## Failure taxonomy

| config | episodes | success | near miss | touched, missed | never contacted | unknown |
|---|---:|---:|---:|---:|---:|---:|
| base__zeroshot | 20 | 0 (0%) | 0 | 7 | 13 | 0 |
| expertonly__12k | 20 | 0 (0%) | 20 | 0 | 0 | 0 |
| expertonly__16k | 20 | 0 (0%) | 11 | 9 | 0 | 0 |
| expertonly__20k | 20 | 1 (5%) | 19 | 0 | 0 | 0 |
| expertonly__4k | 20 | 0 (0%) | 0 | 20 | 0 | 0 |
| expertonly__8k | 20 | 0 (0%) | 0 | 20 | 0 | 0 |

*Near miss* = the block was moved and the best pose overlap reached 0.85 of the 0.95 needed for success (position roughly right, pose not). *Never contacted* = the end-effector never touched the block, which points at reaching/perception rather than manipulation.


## Raw

Per-episode records and full configuration are in the sibling JSON files.
