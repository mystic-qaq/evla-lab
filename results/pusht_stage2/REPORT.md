# Evaluation report

> Historical report generated before the Push-T episode-reset fix. Peak coverage and derived failure categories require re-evaluation. Corrected measurements are in the project README.

| # | config | mode (exec/chunk) | success | 95% Wilson CI | control Hz | p50 lat (ms) | p95 lat (ms) | steps/inference | mean steps | best coverage |
|---|--------|-------------------|---------|---------------|------------|--------------|--------------|-----------------|------------|-----------------|
| 1 | stage2__22k | open_loop (50/50) | **0%** | 0%-16% | 130.0 | 255.02 | 258.75 | 50.0 | 300 | 0.91 |
| 2 | stage2__25k | open_loop (50/50) | **0%** | 0%-16% | 129.2 | 255.87 | 268.44 | 50.0 | 300 | 0.83 |

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

## Failure taxonomy

| config | episodes | success | near miss | touched, missed | never contacted | unknown |
|---|---:|---:|---:|---:|---:|---:|
| stage2__22k | 20 | 0 (0%) | 20 | 0 | 0 | 0 |
| stage2__25k | 20 | 0 (0%) | 13 | 7 | 0 | 0 |

*Near miss* = the block was moved and the best pose overlap reached 0.85 of the 0.95 needed for success (position roughly right, pose not). *Never contacted* = the end-effector never touched the block, which points at reaching/perception rather than manipulation.


## Raw

Per-episode records and full configuration are in the sibling JSON files.
