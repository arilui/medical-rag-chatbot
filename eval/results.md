# Retrieval Evaluation

Evaluated 30 in-scope questions (6 per indexed page) and 12 out-of-scope questions (6 uncovered medical topics, 6 non-medical) using `text-embedding-3-small` and top-5 retrieval.

## Results at the deployed threshold (0.37)

| Metric | Result |
|---|---|
| In-scope questions answered | 100% |
| Hit@1 (top chunk from the correct page) | 100% |
| Recall@5 (correct page in top 5) | 100% |
| Mean reciprocal rank | 1.00 |
| Out-of-scope questions refused | 100% |
| &nbsp;&nbsp;Uncovered medical topics refused | 100% |
| &nbsp;&nbsp;Non-medical questions refused | 100% |

## Score separation

- Lowest top score for an in-scope question: **0.397** (What happens during a panic attack?)
- Highest top score for an out-of-scope question: **0.343** (What triggers an asthma attack?)

When the lowest in-scope score sits above the highest out-of-scope score, a single threshold can separate them perfectly.

## Threshold sweep

| Threshold | Answered | Recall@5 | Refused (all) | Refused (medical) | Refused (non-medical) | Balanced |
|---|---|---|---|---|---|---|
| 0.150 | 100% | 100% | 42% | 0% | 83% | 71% |
| 0.175 | 100% | 100% | 50% | 0% | 100% | 75% |
| 0.200 | 100% | 100% | 50% | 0% | 100% | 75% |
| 0.225 | 100% | 100% | 75% | 50% | 100% | 88% |
| 0.250 | 100% | 100% | 75% | 50% | 100% | 88% |
| 0.275 | 100% | 100% | 83% | 67% | 100% | 92% |
| 0.300 | 100% | 100% | 83% | 67% | 100% | 92% |
| 0.325 | 100% | 100% | 92% | 83% | 100% | 96% |
| 0.350 ◀ best | 100% | 100% | 100% | 100% | 100% | 100% |
| 0.375 | 100% | 100% | 100% | 100% | 100% | 100% |
| 0.400 | 97% | 97% | 100% | 100% | 100% | 98% |
| 0.425 | 97% | 97% | 100% | 100% | 100% | 98% |
| 0.450 | 97% | 97% | 100% | 100% | 100% | 98% |
| 0.475 | 97% | 97% | 100% | 100% | 100% | 98% |
| 0.500 | 87% | 87% | 100% | 100% | 100% | 93% |

## Failures at 0.37

None.
