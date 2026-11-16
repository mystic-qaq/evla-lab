# Leaderboard

Reference config: **ft__bf16__compile__flow4__ope20**

| rank | config | success | 95% CI | Δ vs ref | wins/losses | McNemar p | control Hz |
|------|--------|---------|--------|----------|-------------|-----------|------------|
| 1 | ft__bf16__compile__flow4__ope20 | 10.0% | 4%-23% | +0.0% | 0/0 | 1.000 | 184.7 |
| 2 | ft__bf16__compile__flow4__ope50 | 10.0% | 4%-23% | +0.0% | 3/3 | 1.000 | 269.9 |
| 3 | ft__bf16__compile__ope50 | 7.5% | 3%-20% | -2.5% | 2/3 | 1.000 | 228.7 |
| 4 | ft__bf16__compile__flow4__ens10 | 5.0% | 1%-16% | -5.0% | 1/3 | 0.625 | 17.9 |
| 5 | ft__fp32__ope50 | 5.0% | 1%-16% | -5.0% | 2/4 | 0.688 | 130.4 |
| 6 | ft__bf16__ope50 | 2.5% | 0%-13% | -7.5% | 1/4 | 0.375 | 127.4 |
| 7 | ft__bf16__compile__flow4__receding | 0.0% | 0%-16% | -10.0% | 0/2 | 0.500 | 18.0 |
| 8 | zeroshot__ope50 | 0.0% | 0%-9% | -10.0% | 0/4 | 0.125 | 136.7 |

* marks p < 0.05.

Paired McNemar test over identical seeds; delta is absolute success-rate change.

