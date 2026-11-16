# Decision threshold per benchmark

Threshold tuning of Section "Matching using Development Data": the decision threshold t of each decision model is tuned on the Sol-labeled 2,000 development sets, either per benchmark (bench.) or once per model over all eight benchmarks (model), and applied unchanged to the three test runs with DP. The table gives the test F1 difference (points) to t = 0.5, mean over three runs, per benchmark and decision model, and the test-optimal threshold per benchmark (opt.) as an upper bound. Computed by `src/thresholds_solset_eval.py`.

| Benchmark | Jev 1.13 bench. | Jev 1.13 model | Jev 1.13 opt. | PPLX v1.2 bench. | PPLX v1.2 model | PPLX v1.2 opt. | Clef bench. | Clef model | Clef opt. | Clef-flash bench. | Clef-flash model | Clef-flash opt. | Kev bench. | Kev model | Kev opt. | GPT-6 Luna Dec. bench. | GPT-6 Luna Dec. model | GPT-6 Luna Dec. opt. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Abt-Buy | -0.06 | -0.07 | +0.48 | -0.04 | -0.13 | +0.78 | -2.03 | -1.70 | +1.87 | +1.00 | +1.00 | +2.84 | +1.03 | +1.03 | +1.74 | +1.63 | +2.75 | +2.97 |
| Walmart-Amazon | -0.26 | +0.10 | +0.76 | -1.08 | -1.08 | +0.59 | +0.81 | +1.11 | +1.29 | +11.15 | +11.56 | +15.98 | +5.32 | +6.68 | +7.16 | +2.95 | +5.24 | +5.24 |
| WDC Products | +1.30 | +0.26 | +2.43 | -1.65 | -0.23 | +1.22 | -2.45 | -3.08 | +1.09 | +1.62 | +2.60 | +4.26 | +0.77 | +1.17 | +2.10 | -2.10 | -3.66 | +0.92 |
| DBLP-ACM | +0.71 | -0.07 | +0.81 | -0.10 | +0.04 | +0.95 | +0.32 | +0.64 | +1.07 | +1.74 | -0.11 | +2.09 | +5.56 | -4.96 | +6.19 | +0.77 | +0.77 | +1.48 |
| DBLP-Scholar | -1.18 | +0.07 | +0.15 | -0.28 | -0.04 | +0.22 | -0.67 | -0.62 | +1.46 | -3.03 | +0.97 | +2.76 | -2.08 | -1.51 | +0.51 | +0.23 | +0.23 | +0.67 |
| Dn7 Walmart-Amazon | -0.15 | -0.15 | +0.63 | +1.76 | +0.87 | +3.44 | -1.53 | +0.26 | +0.93 | +15.30 | +15.30 | +19.48 | -0.03 | -2.60 | +1.08 | +2.47 | +2.47 | +3.61 |
| billiger.de 100un | +0.35 | +0.45 | +2.04 | +0.69 | +0.39 | +2.06 | -0.13 | +0.64 | +0.82 | +13.15 | +13.02 | +13.75 | +6.51 | +6.21 | +9.33 | +3.04 | +5.29 | +5.35 |
| Semi-HETER | +2.98 | +0.73 | +5.19 | +0.82 | -0.23 | +1.73 | -1.06 | -2.64 | +0.77 | -4.50 | -1.07 | +1.40 | -4.47 | -4.25 | +0.93 | -0.46 | -0.12 | +0.65 |
| Mean | +0.46 | +0.16 | +1.56 | +0.02 | -0.05 | +1.37 | -0.84 | -0.67 | +1.16 | +4.56 | +5.41 | +7.82 | +1.58 | +0.22 | +3.63 | +1.07 | +1.62 | +2.61 |

Remaining gap to the test-optimal threshold (mean over the benchmarks, opt. minus bench.):

| | Jev 1.13 | PPLX v1.2 | Clef | Clef-flash | Kev | GPT-6 Luna Dec. |
|---|---:|---:|---:|---:|---:|---:|
| Gap | 1.10 | 1.36 | 2.01 | 3.27 | 2.06 | 1.55 |
