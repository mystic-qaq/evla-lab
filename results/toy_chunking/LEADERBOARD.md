# Leaderboard

Reference config: **open_loop-8**

| rank | config | success | 95% CI | Δ vs ref | wins/losses | McNemar p | control Hz |
|------|--------|---------|--------|----------|-------------|-----------|------------|
| 1 | receding-1 | 76.7% | 59%-88% | +36.7% | 11/0 | 0.001 * | 582.4 |
| 2 | temporal_ens-8 | 60.0% | 42%-75% | +20.0% | 6/0 | 0.031 * | 575.0 |
| 3 | open_loop-8 | 40.0% | 25%-58% | +0.0% | 0/0 | 1.000 | 802.7 |

* marks p < 0.05.

Paired McNemar test over identical seeds; delta is absolute success-rate change.
