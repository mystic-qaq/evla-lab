# Leaderboard

Reference config: **base__zeroshot**

| rank | config | success | 95% CI | Δ vs ref | wins/losses | McNemar p | control Hz |
|------|--------|---------|--------|----------|-------------|-----------|------------|
| 1 | expertonly__20k | 5.0% | 1%-24% | +5.0% | 1/0 | 1.000 | 129.2 |
| 2 | base__zeroshot | 0.0% | 0%-16% | +0.0% | 0/0 | 1.000 | 137.2 |
| 3 | expertonly__12k | 0.0% | 0%-16% | +0.0% | 0/0 | 1.000 | 127.3 |
| 4 | expertonly__16k | 0.0% | 0%-16% | +0.0% | 0/0 | 1.000 | 129.1 |
| 5 | expertonly__4k | 0.0% | 0%-16% | +0.0% | 0/0 | 1.000 | 129.8 |
| 6 | expertonly__8k | 0.0% | 0%-16% | +0.0% | 0/0 | 1.000 | 129.4 |

* marks p < 0.05.

Paired McNemar test over identical seeds; delta is absolute success-rate change.
