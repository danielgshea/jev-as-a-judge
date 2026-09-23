# Jev vs. SemIf

| Metric | Jev | SemIf | Comparison |
|---|---:|---:|---|
| Mean quality-score variance | 0.000018995 | 0.000032345 | SemIf is 1.70× higher |
| 95% CI for mean quality-score variance | 0.000007389–0.000031414 | 0.000011397–0.000057971 | Intervals overlap |
| Maximum quality-score variance | 0.000043208 | 0.000080901 | SemIf is 1.87× higher |
| Mean latency per call | 0.50 s | 0.55 s | SemIf is 0.05 s (10%) slower |

Lower variance and latency are better. Variance is the unbiased sample variance within each of five recorded cases over 100 repetitions; the table reports the mean across cases. Latency is measured across successful evaluator calls.

## Variance

![Mean per-case variance ratios for Jev and SemIf](aaa78b6a-f6be-4542-a125-1941c7a8b5df/variance-ratios.svg)

## Latency

![Mean evaluator latency for Jev and SemIf](aaa78b6a-f6be-4542-a125-1941c7a8b5df/latency-comparison.svg)
