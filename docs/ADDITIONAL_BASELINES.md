# Additional baselines of the development-data section

Two baselines of Section "Matching using Development Data" are not shown in the table of the paper: the six handwritten matching rules of our earlier study, given to PPLX v1.2 and Jev 1.13 as a prompt, and logistic regression trained on the Sol-labeled development sets and on the full benchmark training sets.

Test F1 (%) per benchmark, mean over the available runs, and the difference of the mean to DP of the same decision model (logistic regression: to PPLX v1.2 with DP). Handwritten rules: one run; logistic regression: one deterministic fit.

| Method | Runs | Abt-Buy | Walm.-Am. | WDC | DBLP-ACM | DBLP-Sch. | Dn7 | billiger | Semi-H. | Mean | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPLX v1.2 + handwritten rules | 1 | 92.92 | 86.89 | 87.79 | 97.46 | 90.82 | 74.61 | 62.16 | 95.82 | 86.06 | -2.07 |
| Jev 1.13 + handwritten rules | 1 | 90.57 | 86.00 | 86.38 | 96.72 | 92.00 | 72.97 | 61.42 | 94.87 | 85.12 | -1.31 |
| Logistic regression (200) | 1 | 55.16 | 45.84 | 36.63 | 94.06 | 84.55 | 36.32 | 27.90 | 88.67 | 58.64 | -29.49 |
| Logistic regression (2,000) | 1 | 55.04 | 48.42 | 40.76 | 94.19 | 90.66 | 38.05 | 24.21 | 82.86 | 59.27 | -28.85 |
| Logistic regression (all, benchmark labels) | 1 | 53.63 | 52.49 | 42.77 | 94.76 | 91.43 | 49.36 | 29.41 | 67.21 | 60.13 | -28.00 |

## Handwritten rules

The handwritten-rules prompt (`R_handwritten` in the code) replaces DP with the following instruction.

Product benchmarks:

```text
Your task is to decide if two product descriptions match.

The following rules regarding product features need to be observed:

1. The brand of matching products must be the same if available
2. Model names of matching products must be the same if available
3. Model numbers of matching products must be the same if available
4. Additional features of matching products must be the same if available
5. Matching attributes may not have the exact same surface form due to different case, typos, value formats, etc.
6. If an attribute is missing for one description, it is likely still a match if the existing attributes match.

Do the two product descriptions match?
```

Publication benchmarks:

```text
Your task is to decide if two publications match.

The following rules regarding publication features need to be observed:

1. The authors must be the same.
2. The title must be the same.
3. The venue should be the same but could differ.
4. The year should be the same but could differ.
5. Matching attributes may not have the exact same surface form due to different case, typos, value formats, etc.
6. If an attribute is missing for one publication, it is likely still a match if the existing attributes match.

Do the two publications match?
```
