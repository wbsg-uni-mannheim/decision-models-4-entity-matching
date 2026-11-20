# Similarity baselines of the cascades

Section "Model Cascades" of the paper compares each cascade (t = 0.05) with two baselines that defer other pairs to the same partner: a random baseline and a similarity baseline. Both baselines defer as many pairs per benchmark and run as the cascade, so they differ from the cascade only in which pairs they defer. The repository contains two variants of the similarity baseline. The current version of the paper reports the boundary variant, the first version reported the most-similar variant.

| Baseline | Script | Results |
|---|---|---|
| Random | `src/cascade_random_deferral_eval.py` | `results/cascade_random_deferral_t005_20261107.json` |
| Similarity, boundary variant (current paper) | `src/cascade_similarity_boundary_eval.py` | `results/cascade_similarity_boundary_20261117.json` |
| Similarity, most-similar variant (first paper version) | `src/cascade_similarity_deferral_eval.py` | `results/cascade_similarity_deferral_20261108.json` (variant `high_0.16`) |

Both similarity variants use the same scores, `results/similarity_scores_20261108.json` (computed by `runners/similarity_scores_20261108.py`): the cosine of the two record embeddings (text-embedding-3-small) and the Jaccard similarity of the token sets of the two records. No labels or model outputs enter the scores.

## Boundary variant

1. **Scores:** cosine and Jaccard are kept as two separate lists.
2. **Threshold per benchmark and score:** on the benchmark validation set (without the pairs in `data/<benchmark>/rules_train.json`), with its gold labels. With k the number of validation matches, the threshold t is the score of the k-th most similar validation pair. A similarity matcher at t thus makes as many false positives as false negatives.
3. **Number of deferred pairs:** per decision model, run, and benchmark, as many pairs as the cascade defers.
4. **Order on the test set:** per score, the test pairs are sorted by score (ties by `pair_id`). Starting at t, the nearest unused pair above and the nearest unused pair below t are taken in turn, above first. When one side runs out, the other side continues. The two scores take turns at every pick, cosine first, and a pair already taken is skipped. The baseline defers the first N pairs of this order. There are no random draws.
5. **Comparison:** F1, cost, and the difference to the cascade with the paired bootstrap and the multi-bootstrap over runs, as for the random baseline.

Unlike the cascade, this baseline uses labels (the validation labels for t).

### Thresholds

| Benchmark | Validation pairs | Matches (k) | t cosine | t Jaccard | Match share in band | Match share overall |
|---|---:|---:|---:|---:|---:|---:|
| Abt-Buy | 1,891 | 200 | 0.816 | 0.222 | 21.6% | 10.6% |
| Walmart-Amazon | 2,033 | 189 | 0.884 | 0.444 | 21.4% | 9.3% |
| WDC Products | 4,500 | 500 | 0.784 | 0.229 | 26.7% | 11.1% |
| DBLP-ACM | 2,427 | 431 | 0.894 | 0.562 | 57.3% | 17.8% |
| DBLP-Scholar | 5,682 | 1,061 | 0.822 | 0.391 | 46.0% | 18.7% |
| Dn7 Walmart-Amazon | 8,684 | 153 | 0.917 | 0.517 | 20.0% | 1.8% |
| billiger.de | 4,452 | 485 | 0.869 | 0.263 | 24.4% | 10.9% |
| Semi-HETER | 411 | 154 | 0.846 | 0.328 | 47.2% | 37.5% |

Match share in band: the same selection applied to the validation pairs, with a band as large as the mean share of pairs the cascades defer on that benchmark. A mixed band shows that t lies where matches and non-matches overlap.

## Results

Mean test F1 (%) over the eight benchmarks, 3-run mean. Δ: cascade minus baseline; † marks a difference that is significant under both bootstraps.

| Model | Partner | Cascade | Random | Most-similar variant | Boundary variant | Δ boundary |
|---|---|---:|---:|---:|---:|---:|
| Jev 1.13 | Sol | 90.80 | 87.77 | 88.49 | 89.00 | +1.80† |
| Jev 1.13 | Ensemble | 91.30 | 87.86 | 88.58 | 89.25 | +2.05† |
| PPLX v1.2 | Sol | 90.81 | 88.49 | 89.24 | 89.06 | +1.75† |
| PPLX v1.2 | Ensemble | 91.28 | 88.59 | 89.33 | 89.18 | +2.10† |
| Clef | Sol | 90.81 | 86.97 | 88.06 | 89.06 | +1.75† |
| Clef | Ensemble | 91.28 | 87.05 | 88.14 | 89.27 | +2.01† |
| Luna Dec. | Sol | 90.66 | 86.58 | 87.84 | 87.46 | +3.20† |
| Luna Dec. | Ensemble | 91.12 | 86.65 | 87.92 | 87.57 | +3.54† |

The most-similar variant defers the pairs with the highest mean of cosine and Jaccard, above one global threshold set without labels so that it would defer 16% of the validation pairs. It defers a fixed set of pairs for all models, so its number of deferred pairs differs from that of each cascade.
