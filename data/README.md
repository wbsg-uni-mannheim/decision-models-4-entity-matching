# Data

One folder per benchmark. All JSON files except the development sets hold a list of cases. A case has the fields `benchmark`, `pair_id`, `entity_ids`, `records` (the two records with typed attributes), `label` (1 = match, 0 = non-match), and `input_hash` (SHA-256 of the records).

| Folder | Benchmark | Test pairs (matches) | Validation pairs (matches) |
|---|---|---:|---:|
| `abt-buy` | Abt-Buy | 1,916 (206) | 1,901 (205) |
| `walmart-amazon` | Walmart-Amazon | 2,049 (193) | 2,047 (193) |
| `dblp-acm` | DBLP-ACM | 2,473 (444) | 2,449 (442) |
| `dblp-scholar` | DBLP-Scholar | 5,742 (1,070) | 5,705 (1,070) |
| `wdc` | WDC Products (80 % corner cases, 100 % unseen) | 4,500 (500) | 4,500 (500) |
| `dn7-walmart-amazon` | Dn7 Walmart-Amazon | 8,684 (153) | 8,684 (153) |
| `semi-heter` | Semi-HETER | 414 (159) | 411 (154) |
| `billiger-de` | billiger.de (German, 100 % unseen) | 4,447 (348) | 4,452 (485) |
| | Total | 30,225 | 30,149 |

## Files per benchmark

| File | Content | Labels |
|---|---|---|
| `test.json` | Test split of the benchmark, all rows. | Gold |
| `valid.json` | Validation split of the benchmark. Pairs whose records also occur in the test split, duplicate pairs, and groups of identical pairs with conflicting labels are removed. Used for the prompt and threshold selection on validation data. | Gold |
| `rules_train.json` | 100 validation pairs that an earlier rule-writing step saw. They are left out whenever the validation split is scored. | Gold |
| `escalation.json` | Test pairs that a first routing step with thresholds from the validation split marked as uncertain. The frontier LLMs answered these pairs first. | Gold |
| `reference_rest.json` | The remaining test pairs (test minus `escalation`). With both files, the frontier LLMs answered every test pair, so every cascade in the paper is scored from stored answers. | Gold |
| `escalation_sample.json` | Stratified sample of `escalation.json` for a first small run. | Gold |
| `solset.json` | Union of the 200-pair and the 2,000-pair development set in case form. Used to tune thresholds on development data. | GPT-6.1 Sol |
| `solvalid2000.json` | The 20 % split of the 2,000-pair development set in case form. | GPT-6.1 Sol |
| `development/set{200,2000}_{all,train,valid}.json.gz` | The development sets (JSON lines) and their 80/20 split into a part for demonstrations and rules (`train`) and a part for checking the rules (`valid`). | GPT-6.1 Sol |

The development sets contain record pairs of the benchmark that are neither in the validation nor in the test split. The pairs were selected by similarity search and active learning as described in Steiner and Bizer (2026, [arXiv:2606.28823](https://arxiv.org/abs/2606.28823)) and labeled by GPT-6.1 Sol with the default prompt DP. The `label` field of these files is the Sol label, not a gold label. The code expects the development sets under `runs/labeled_chapter_20261028/sets/<benchmark>/`:

```bash
mkdir -p runs/labeled_chapter_20261028/sets && for b in data/*/; do b=$(basename "$b"); mkdir -p runs/labeled_chapter_20261028/sets/$b && cp data/$b/development/*.json.gz runs/labeled_chapter_20261028/sets/$b/; done
```

## Sources and licenses

We redistribute the record pairs of the benchmarks in a converted form. Please cite the original benchmarks and respect their terms.

| Benchmark | Source | License stated by the source |
|---|---|---|
| Abt-Buy, DBLP-ACM, DBLP-Scholar | Köpcke, Thor, and Rahm (2010), [Leipzig benchmark datasets](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution); splits of DeepMatcher (Mudgal et al., 2018) | CC BY 4.0 |
| Walmart-Amazon | [Magellan data repository](https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository); splits of DeepMatcher (Mudgal et al., 2018) | No license stated |
| WDC Products | Peeters, Der, and Bizer (2024), [WDC Products](https://webdatacommons.org/largescaleproductcorpus/wdc-products/) | Repository: BSD-3-Clause; no separate data license stated |
| Dn7 Walmart-Amazon | Papadakis et al. (2024), A Critical Re-evaluation of Record Linkage Benchmarks for Learning-Based Matching Algorithms, [DLMatchers](https://github.com/gpapadis/DLMatchers) | No license stated |
| Semi-HETER | Wang, Li, and Hirota (2021), [Machamp](https://github.com/megagonlabs/machamp) | BSD-3-Clause |
| billiger.de | Steiner et al. (2026), [billiger.de Products](https://github.com/wbsg-uni-mannheim/billiger-de-products) | No license stated |

The records of WDC Products come from public web pages and contain, in a few offers, contact e-mail addresses of shops.
