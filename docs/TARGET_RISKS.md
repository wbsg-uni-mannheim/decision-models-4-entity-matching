# Target risks of the cascade

Cascade of each decision model (selected prompt, F0) at each target risk. A pair is deferred if min(p, 1-p) > t, where t is the largest threshold whose error among the accepted pairs of the full validation sets stays at or below the target risk (`valsel_eval.tune_risk()`). Deferred pairs go either to GPT-6.1 Sol alone (+ Sol) or to the majority vote of GPT-6.1 Sol, Claude Opus 5.5, and Gemini 4 Argon (+ Ensemble). All other columns are measured on the 30,225 test pairs. Selective risk: error rate among the pairs the decision model decides alone. Cost: the decision model on all test pairs plus the partner on the deferred pairs, at standard list prices (Gemini 4 Argon ran on the flex tier at half price; its cost is doubled here). These are the values of the paper (Section "Match Probabilities and Cascade").

| Model | Target risk | t | Deferred (%) | Selective risk (%) | Partner | Mean F1 | Errors | Cost (USD) |
|---|---|---|---|---|---|---|---|---|
| Jev 1.13 | 0.25% | 0.080 | 16.7 | 0.11 | Alone | 88.06 | 701 | 0.97 |
| | | | | | + Sol | 90.68 | 586 | 6.65 |
| | | | | | + Ensemble | 91.02 | 545 | 33.95 |
| Jev 1.13 | 0.50% | 0.120 | 11.7 | 0.28 | Alone | 88.06 | 701 | 0.97 |
| | | | | | + Sol | 90.70 | 582 | 5.24 |
| | | | | | + Ensemble | 91.01 | 545 | 25.90 |
| Jev 1.13 | 1.00% | 0.230 | 6.1 | 0.72 | Alone | 88.06 | 701 | 0.97 |
| | | | | | + Sol | 90.46 | 589 | 3.41 |
| | | | | | + Ensemble | 90.71 | 557 | 15.44 |
| PPLX v1.2 | 0.25% | 0.030 | 16.6 | 0.19 | Alone | 89.13 | 641 | 0.27 |
| | | | | | + Sol | 90.68 | 586 | 5.78 |
| | | | | | + Ensemble | 91.06 | 544 | 32.13 |
| PPLX v1.2 | 0.50% | 0.072 | 10.3 | 0.28 | Alone | 89.13 | 641 | 0.27 |
| | | | | | + Sol | 90.69 | 584 | 4.09 |
| | | | | | + Ensemble | 91.07 | 542 | 22.49 |
| PPLX v1.2 | 1.00% | 0.194 | 5.0 | 0.75 | Alone | 89.13 | 641 | 0.27 |
| | | | | | + Sol | 90.58 | 587 | 2.41 |
| | | | | | + Ensemble | 90.91 | 549 | 12.89 |
| Clef | 0.50% | 0.092 | 11.4 | 0.32 | Alone | 87.70 | 704 | 3.82 |
| | | | | | + Sol | 90.58 | 582 | 7.98 |
| | | | | | + Ensemble | 90.88 | 545 | 28.04 |
| Luna Dec. | 0.50% | 0.050 | 12.5 | 0.37 | Alone | 87.00 | 815 | 1.52 |
| | | | | | + Sol | 90.59 | 583 | 5.93 |
| | | | | | + Ensemble | 90.97 | 543 | 27.16 |

## F1 per benchmark

| Model | Target risk | Partner | Abt-Buy | Walmart-Amazon | WDC | DBLP-ACM | DBLP-Scholar | Dn7 W-A | billiger.de | Semi-HETER |
|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.13 | 0.25% | Alone | 91.60 | 88.49 | 87.14 | 98.11 | 92.78 | 77.81 | 73.06 | 95.48 |
|  |  | + Sol | 96.44 | 94.18 | 91.54 | 97.80 | 92.78 | 79.58 | 77.57 | 95.51 |
|  |  | + Ensemble | 96.21 | 93.43 | 91.42 | 98.01 | 93.64 | 81.18 | 78.71 | 95.51 |
| Jev 1.13 | 0.50% | Alone | 91.60 | 88.49 | 87.14 | 98.11 | 92.78 | 77.81 | 73.06 | 95.48 |
|  |  | + Sol | 95.94 | 94.18 | 91.24 | 97.91 | 92.83 | 79.79 | 78.22 | 95.51 |
|  |  | + Ensemble | 95.96 | 93.43 | 91.11 | 98.01 | 93.65 | 81.18 | 79.19 | 95.51 |
| Jev 1.13 | 1.00% | Alone | 91.60 | 88.49 | 87.14 | 98.11 | 92.78 | 77.81 | 73.06 | 95.48 |
|  |  | + Sol | 95.90 | 93.64 | 90.44 | 97.79 | 93.11 | 79.78 | 77.83 | 95.18 |
|  |  | + Ensemble | 95.90 | 93.13 | 90.04 | 97.90 | 93.85 | 80.77 | 78.89 | 95.18 |
| PPLX v1.2 | 0.25% | Alone | 94.76 | 90.44 | 87.86 | 98.33 | 93.63 | 77.69 | 74.49 | 95.82 |
|  |  | + Sol | 96.44 | 93.94 | 91.54 | 97.80 | 92.78 | 79.69 | 77.71 | 95.51 |
|  |  | + Ensemble | 96.21 | 93.43 | 91.42 | 98.01 | 93.64 | 81.50 | 78.76 | 95.51 |
| PPLX v1.2 | 0.50% | Alone | 94.76 | 90.44 | 87.86 | 98.33 | 93.63 | 77.69 | 74.49 | 95.82 |
|  |  | + Sol | 96.44 | 94.18 | 91.65 | 97.80 | 92.82 | 79.37 | 77.71 | 95.51 |
|  |  | + Ensemble | 96.44 | 93.43 | 91.53 | 98.01 | 93.68 | 81.18 | 78.76 | 95.51 |
| PPLX v1.2 | 1.00% | Alone | 94.76 | 90.44 | 87.86 | 98.33 | 93.63 | 77.69 | 74.49 | 95.82 |
|  |  | + Sol | 96.17 | 93.67 | 91.36 | 98.01 | 92.94 | 79.58 | 77.39 | 95.51 |
|  |  | + Ensemble | 96.17 | 93.16 | 91.02 | 98.12 | 93.81 | 81.18 | 78.28 | 95.51 |
| Clef | 0.50% | Alone | 92.38 | 88.02 | 85.77 | 97.66 | 92.50 | 75.47 | 74.93 | 94.84 |
|  |  | + Sol | 96.67 | 93.37 | 91.39 | 97.80 | 93.04 | 78.95 | 77.93 | 95.51 |
|  |  | + Ensemble | 96.19 | 92.62 | 91.38 | 98.01 | 93.78 | 80.54 | 78.99 | 95.51 |
| Luna Dec. | 0.50% | Alone | 93.93 | 85.25 | 86.96 | 98.00 | 92.33 | 75.26 | 69.14 | 95.12 |
|  |  | + Sol | 96.44 | 93.94 | 91.09 | 97.69 | 93.32 | 79.37 | 77.62 | 95.24 |
|  |  | + Ensemble | 96.21 | 93.43 | 91.09 | 97.91 | 94.05 | 81.18 | 78.67 | 95.24 |

## Reference LLMs alone on all test pairs (list prices)

Gemini 4 Argon ran on the flex tier and Claude Opus 5.5 answered the non-deferred pairs through the Batch API, both at half price, and 2,499 requests of GPT-6.1 Sol ran on the flex tier; all of these costs are doubled here.

| Model | Mean F1 | Cost (USD) |
|---|---|---|
| GPT-6.1 Sol | 90.64 | 22.76 |
| Claude Opus 5.5 | 90.72 | 82.07 |
| Gemini 4 Argon | 90.95 | 38.15 |
| Majority of the three | 91.06 | 142.98 |
