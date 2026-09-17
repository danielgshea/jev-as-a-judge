# Judge reliability experiment

## Result

The 10-repetition experiment is recorded in [LangSmith](https://smith.langchain.com/o/fd6b1198-8e6a-4f06-80f5-20e1b40ded12/datasets/fc68427d-1695-49eb-901b-fb4c733c24bc/compare?selectedSessions=2bc3080f-eb5e-469a-b60a-8951ec5ecd46).

The experiment froze one weather-agent output for each of the five dataset cases, then repeated only the judge calls. This isolates judge variability from agent and web-search variability. It ran six evaluators on every repetition:

- Jev quality (`Noul`), score (`Score`), and outcome (`Choice`)
- LLM quality, score, and outcome using `gpt-5.6-luna`

## Statistical method

For the numeric quality and score metrics, variance is the unbiased sample variance across the 10 repeated judgments within each case. Reported variance is the mean of those five case-level variances.

Confidence intervals use 10,000 bootstrap resamples of the five frozen cases. The bootstrap unit is the case, not an individual judge call, so the intervals reflect the limited prompt sample. Choice reliability is reported as disagreement with the modal choice within each case.

## Results

| Metric | Judge | Mean | 95% CI for mean | Mean case variance | 95% CI for variance | Std. dev. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Quality | Jev | 0.9088 | [0.8607, 0.9440] | 0.0000106 | [0.0000045, 0.0000205] | 0.0474 |
| Quality | LLM | 0.9830 | [0.9657, 0.9957] | 0.0025455 | [0.0000086, 0.0076026] | 0.0514 |
| Score | Jev | 0.6288 | [0.2867, 0.9472] | 0.0000612 | [0.0000268, 0.0000943] | 0.3862 |
| Score | LLM | 0.8800 | [0.6800, 1.0000] | 0.0088889 | [0.0000000, 0.0266667] | 0.2157 |

### Variance comparison

| Metric | LLM variance / Jev variance | 95% bootstrap CI | LLM − Jev variance |
| --- | ---: | ---: | ---: |
| Quality | 239.75× | [1.80×, 379.09×] | 0.002535 |
| Score | 145.19× | [0.00×, 445.68×] | 0.008828 |

The LLM judge showed substantially higher point-estimate variance for both numeric metrics. The quality variance difference has a bootstrap interval entirely above zero. The score variance interval includes zero, so this five-case experiment does not establish a statistically clear score-variance difference despite the much larger point estimate.

The LLM also gave higher average scores: +0.0742 quality points (95% CI [0.0481, 0.1174]) and +0.2512 normalized score points (95% CI [0.0528, 0.4718]). That is a calibration difference, not evidence that the LLM judge is more accurate.

Both judges had 0% modal-choice disagreement on these five cases. The categorical outcome task was therefore perfectly stable in this sample, but it was less sensitive to variability than the numeric judgments.

## Interpretation and limitations

For this weather-agent dataset, Jev was the more stable numeric judge: its repeated quality and score outputs moved much less across identical inputs. The LLM judge was consistently more generous and more variable. Both judges were stable on the coarse three-way outcome classification.

The main limitation is the five-case dataset. Bootstrap intervals are consequently wide, especially for variance ratios, and the results should be treated as evidence about this evaluator setup rather than a general claim about Jev or LLM judges. More distinct cases—especially borderline and clearly incorrect answers—would make the reliability comparison stronger.

Reproduce the remote experiment with:

```bash
uv run python src/evals/judge_reliability.py --trials 10
```

Run the same frozen-case workflow without uploading to LangSmith with:

```bash
uv run python src/evals/judge_reliability.py --local --trials 10
```
