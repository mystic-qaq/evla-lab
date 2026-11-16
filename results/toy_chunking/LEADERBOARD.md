# Leaderboard

Reference config: **receding-1**

| rank | config | success | 95% CI | Δ vs ref | wins/losses | McNemar p | control Hz |
|------|--------|---------|--------|----------|-------------|-----------|------------|
| 1 | receding-1 | 76.7% | 59%-88% | +0.0% | 0/0 | 1.000 | 582.4 |
| 2 | temporal_ens-8 | 60.0% | 42%-75% | -16.7% | 0/5 | 0.062 | 575.0 |
| 3 | open_loop-8 | 40.0% | 25%-58% | -36.7% | 0/11 | 0.001 * | 802.7 |

* marks p < 0.05.

Paired McNemar test over identical seeds; delta is absolute success-rate change.
