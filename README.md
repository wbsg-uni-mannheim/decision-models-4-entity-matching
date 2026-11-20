# Entity Matching using Decision Models

This repository contains the code, prompts, configurations, data, and aggregated results of the paper *Entity Matching using Decision Models*.

Decision models are pre-trained language models that return probabilities for predefined answer options instead of generated text. The paper evaluates six decision models on eight entity matching benchmarks in three settings: alone without task-specific training data, in model cascades with frontier LLMs, and with development sets of record pairs labeled by a frontier LLM.

## Models

| Paper name | Identifier | Access |
|---|---|---|
| Jev | `typesafe/jev-1.13` | OpenRouter Decisions API |
| PPLX | `perplexity/pplx-decider-v1.2` | OpenRouter Decisions API; local runs with the open weights |
| Clef | `cloudflare/clef` | OpenRouter Decisions API |
| Clef-flash | `cloudflare/clef-flash` | OpenRouter Decisions API |
| Kev | `jaredpalmer/kev-4b` | OpenRouter Decisions API |
| GPT-6 Luna Dec. | `openai/gpt-6-luna-decisions` | OpenRouter Decisions API |
| GPT-6 Luna Gen. | `openai/gpt-6-luna` | OpenRouter Chat Completions |
| Qwen3.8-27B | `qwen/qwen3.8-27b` | OpenRouter Chat Completions; local runs with the open weights |
| GPT-6.1 Sol (frontier LLM) | `gpt-6.1-sol` | OpenAI Chat Completions |
| Claude Opus 5.5 (frontier LLM) | `anthropic/claude-opus-5.5` | OpenRouter |
| Gemini 4 Argon (frontier LLM) | `google/gemini-4-argon` | OpenRouter |
| GPT-6 Astra | `gpt-6-astra` | writes the matching rules from the development sets |

All API runs took place between 14 October and 5 November 2026. The exact request bodies, model identifiers, and provider settings of every run are frozen in the manifests in `configs/`. The paper reports all costs at the providers' standard list prices.

## Repository layout

| Folder | Content |
|---|---|
| `src/` | Shared code: request format (`protocol.py`, `prompt12.py`), API client (`run.py`), data preparation (`prepare.py`), development sets and rules (`labeled_chapter.py`, `solset_b0.py`, `solvalid_rulesfs.py`), cascades (`escalation.py`, `reference_rest.py`), and the evaluation scripts (`*_eval.py`) that compute the numbers of the paper, including the bootstrap tests (`multibootstrap_eval.py`). |
| `runners/` | Scripts that sent the API requests of each study. Each runner checks a frozen manifest before it sends a request. `runners/local_throughput/` holds the local inference runs. |
| `configs/` | Frozen manifests of every study, the development-set definition, and the matching rules written by GPT-6 Astra. |
| `data/` | Test and validation splits of the eight benchmarks and the development sets labeled by GPT-6.1 Sol. See [`data/README.md`](data/README.md) for the files, sources, and licenses. |
| `results/` | Aggregated results (F1, costs, thresholds, cascades, bootstrap intervals) that the tables of the paper are built from. |
| `docs/` | Documentation of the prompts, rules, thresholds, and further details (see below). |

## Documentation

| File | Content |
|---|---|
| [`docs/PROMPTS.md`](docs/PROMPTS.md) | The default prompt DP and the nine other prompts of the prompt study, with their test F1 per model |
| [`docs/INPUT_VARIANTS.md`](docs/INPUT_VARIANTS.md) | One record pair in all serializations |
| [`docs/RULES.md`](docs/RULES.md) | The matching rules that GPT-6 Astra derived from the development sets |
| [`docs/THRESHOLDS.md`](docs/THRESHOLDS.md) | Decision and deferral thresholds per model and benchmark |
| [`docs/BENCHMARK_ATTRIBUTES.md`](docs/BENCHMARK_ATTRIBUTES.md) | Attributes per benchmark that enter the serialization |
| [`docs/ADDITIONAL_BASELINES.md`](docs/ADDITIONAL_BASELINES.md) | Handwritten rules and logistic regression baselines |
| [`docs/CASCADE_BASELINES.md`](docs/CASCADE_BASELINES.md) | Random and similarity baselines of the model cascades, including both variants of the similarity baseline |
| [`docs/LOCAL_INFERENCE.md`](docs/LOCAL_INFERENCE.md) | Hardware, software, and settings of the local runs |

## Names in the code

The code keeps the names we used during the experiments. The main differences to the paper:

| Paper | Code |
|---|---|
| Default prompt DP | `B0` |
| Prompt study (ten prompts) | `prompt12` (two further rule prompts of the study moved to the development-set experiments) |
| Per model and benchmark, the prompt chosen in an earlier version of the prompt study (used for the first routing step, see [`data/README.md`](data/README.md)) | `F0` |
| Development sets, demonstrations, rules (Section "Matching using Development Data") | `labeled_chapter`, `labeledch`, `fewshot`, `rules`, `rulesfs` |
| Development sets labeled by GPT-6.1 Sol | `solset`, `solvalid2000` |
| Frontier LLMs Sol, Opus, Argon | `sol61`, `opus55`, `argon4` (also `sota`) |
| PPLX (v1.2) | `pplx_v12`; `protocol.py` still lists the earlier version v1.1 under `pplx` |
| Luna Dec., Luna Gen. | `luna_decisions`, `luna` |

## Running the code

Python 3.12 with `numpy`; `matplotlib` for figures. The local runs need the environments described in [`docs/LOCAL_INFERENCE.md`](docs/LOCAL_INFERENCE.md). Run scripts from the repository root with `PYTHONPATH=src`, for example:

```bash
PYTHONPATH=src python src/prompt12_runs123_eval.py --help
```

- **Paths:** Internal paths, host names, and user names are replaced by placeholders such as `/path/to/decision-em` (the repository root) and `/path/to/hf_cache`. Replace them with your own paths. The SHA-256 hashes of source files in the frozen manifests refer to the files before this replacement.
- **API keys:** `run.py` reads `OPENROUTER_API_KEY` from the environment or from `.secrets/source.env`, and the runners that call the OpenAI API directly (GPT-6.1 Sol, GPT-6 Astra) read `OPENAI_API_KEY`. The runners also keep a local budget ledger in `artifacts/`.
- **Raw responses:** Most evaluation scripts read the raw per-pair responses from `runs/`. They are available as release assets (see below); `MANIFEST.md` there says where each folder belongs under `runs/`. The aggregated numbers are in `results/`.
- **Ditto:** Ditto is trained and run with the code of our earlier labeling study, public at [wbsg-uni-mannheim/Automatic-data-labeling](https://github.com/wbsg-uni-mannheim/Automatic-data-labeling) (`third_party/ditto_modern`), with the settings named there and in our configs. The checkpoints are not included.
- **Development sets:** The code reads them from `runs/labeled_chapter_20261028/sets/`; [`data/README.md`](data/README.md) shows how to copy them there.

## Raw per-pair results

The per-pair answers of all runs are available as release assets of the release [`raw-results-v1`](https://github.com/wbsg-uni-mannheim/decision-models-4-entity-matching/releases/tag/raw-results-v1), together with `MANIFEST.md` (content of each archive and the paper section it covers) and `SHA256SUMS`. Each answer line holds the prediction, the match probability (decision models), and every attempt with the raw provider response, the model version, token usage, cost, and time. Account identifiers in provider responses are replaced by `redacted`.

| Archive | Content | Size |
|---|---|---:|
| `zeroshot_dp_runs123.tar.gz` | DP on all test pairs, runs 1 to 3, decision models and GPT-6 Luna Gen. | 154 MB |
| `frontier_llms_runs123.tar.gz` | GPT-6.1 Sol, Claude Opus 5.5, Gemini 4 Argon on all test pairs, runs 1 to 3 | 84 MB |
| `qwen38_api_runs123.tar.gz` | Qwen3.8-27B with DP via the API, runs 1 to 3 | 15 MB |
| `prompt_study_test_runs123.tar.gz` | Prompt study on all test pairs, runs 1 to 3, with the API throughput measurements | 956 MB |
| `prompt_and_threshold_validation.tar.gz` | Prompts and the first routing step on the validation splits | 355 MB |
| `input_variants_runs123.tar.gz` | Serializations and perturbations, runs 1 to 3 | 1,133 MB |
| `development_sets_sol_labels.tar.gz` | Selection and GPT-6.1 Sol labels of the development sets, decision models with DP on them | 23 MB |
| `development_sets_demonstrations_rules.tar.gz` | Demonstrations, rules, and both with PPLX and Jev, runs 1 to 3 | 264 MB |
| `rule_derivation_gpt6_astra.tar.gz` | Rule derivation by GPT-6 Astra, three runs | 16 MB |
| `ditto_logistic_regression.tar.gz` | Ditto predictions (three seeds, no checkpoints) and logistic regression | 51 MB |
| `local_inference_full_runs123.tar.gz` | Local runs of Ditto, PPLX v1.2, Qwen3.8-27B, and GLM-5.3-Flash on all test pairs, with timings | 15 MB |

## License

The code is released under the MIT License (see [`LICENSE`](LICENSE)). The benchmark data keep the licenses of their sources, which are listed in [`data/README.md`](data/README.md).

## Citation

See [`CITATION.cff`](CITATION.cff).
