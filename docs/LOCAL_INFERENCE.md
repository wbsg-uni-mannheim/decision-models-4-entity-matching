# Local Inference

This file describes the runs with open-weight models on our own GPUs: the throughput comparison and the local Qwen3.8-27B runs. No API calls are made in these runs. The scripts are in [`runners/local_throughput/`](../runners/local_throughput/), the aggregated results in `results/local_throughput_20261103.json` (sample of 9,164 test pairs) and `results/throughput_local_full_20261107.json` (all 30,225 test pairs, three runs).

## Hardware

- NVIDIA H100 NVL (94 GB) GPUs on a Slurm cluster.
- One GPU for Ditto, PPLX v1.2, and Qwen3.8-27B; four GPUs (tensor parallel) for GLM-5.3-Flash.

## Software

[`setup_envs.sh`](../runners/local_throughput/setup_envs.sh) creates two Python 3.12 environments with `uv` (we used Python 3.12.13). The versions below are those installed at the time of the runs:

| Environment | Packages | Used for |
|---|---|---|
| `pplx` | torch 2.14.0, transformers 5.17.0, flash-linear-attention 0.5.2, tokenizers 0.23.3, triton 3.8.0, safetensors 0.8.0, huggingface-hub 1.31.0, numpy 2.5.4 | Ditto, model download, summary report |
| `vllm` | vLLM 0.33.0 with torch 2.14.0, transformers 5.20.2, flashinfer 0.7.2.post1 (CUDA 13), triton 3.8.0, tokenizers 0.23.3, huggingface-hub 2.4.0, numpy 2.3.5 | PPLX v1.2, Qwen3.8-27B, GLM-5.3-Flash |

Model revisions (Hugging Face snapshots): `perplexity-ai/pplx-decider-v1.2` cce0f3f, `Qwen/Qwen3.8-27B` 1d4bf0f, `zai-org/GLM-5.3-Flash` 46a050b.

The script imports `protocol.py`, `prompt12.py`, and `qwen38_baseline.py` from `src/`. The PPLX prompt encoder (`pplx_decider`, version 0.3.0) is not a pip package; it ships with the checkpoint under `source/src/pplx_decider` of the model snapshot and must be on the Python path.

PPLX v1.2 (`perplexity-ai/pplx-decider-v1.2`) runs in vLLM through the plugin in [`pplx_vllm_plugin/`](../runners/local_throughput/pplx_vllm_plugin/). It reads the decision at each answer marker of the prompt instead of at the last token. Install it into the `vllm` environment:

```bash
uv pip install --python <vllm-env>/bin/python ./runners/local_throughput/pplx_vllm_plugin
```

[`download.sbatch`](../runners/local_throughput/download.sbatch) downloads the weights of `perplexity-ai/pplx-decider-v1.2`, `Qwen/Qwen3.8-27B`, and `roberta-base` into the Hugging Face cache.

## Settings per model

All models get the DP prompt and the default serialization of the paper. Times are wall-clock seconds from record text to decision, including tokenization, excluding model loading and compilation.

| Model | Weights | Engine | Batched setting |
|---|---|---|---|
| Ditto (RoBERTa-base) | fp16 | PyTorch | batch size 1,024, length-sorted batches per benchmark |
| PPLX v1.2 | bf16 | vLLM pooling runner with the plugin | `max_num_seqs` 128, `max_num_batched_tokens` 32,768, `max_model_len` 8,192 |
| Qwen3.8-27B | bf16 | vLLM `LLM.chat` | thinking off, JSON schema with one Boolean field, `max_tokens` 64, `max_num_seqs` 512, `max_model_len` 16,384, the model's default generation settings (`--sampling api`), seed 0, 1, 2 for the three runs |
| GLM-5.3-Flash | native FP8 weights of the release | vLLM `LLM.chat`, tensor parallel 4 | reasoning effort `low` (thinking cannot be switched off), temperature 0, `max_tokens` 1,024, `max_num_seqs` 64, `max_model_len` 8,192 |

Each model also runs with batch size 1 (one request after the other). Ditto uses the checkpoints trained on the 2,000-pair development sets with the seeds 42, 52, and 62 (one per run). The Ditto code (`ditto_modern`) comes from our labeling study (Steiner and Bizer, 2026, [arXiv:2606.28823](https://arxiv.org/abs/2606.28823)); the script expects it under `/path/to/Automatic-data-labeling/third_party`. The trained checkpoints are not part of this repository.

## Running

The four `full_*.sbatch` files run the three repetitions on all test pairs (`LT_FULL=1`). Replace the `/path/to/...` placeholders with your own paths first:

- `/path/to/decision-em`: the root of this repository
- `/path/to/lt_envs`: the folder with the two environments
- `/path/to/hf_cache`: the Hugging Face cache

The `cd` line of the sbatch files must point to the folder that holds `local_throughput_20261103.py` (in this repository `runners/local_throughput`).

```bash
sbatch runners/local_throughput/full_pplx.sbatch
```

The GLM job also sets CUDA 13.3 paths and several vLLM variables (timeouts, FlashInfer and all-reduce switches); see `full_glm.sbatch`. Each PPLX call writes a batch-size-1 run and a batched run. The 200 warm-up pairs come from `sample.json`, which `python local_throughput_20261103.py sample` writes with a fixed seed.

After the four jobs, `LT_FULL=1 LT_OUT=<output folder> python local_throughput_20261103.py report_full` (in the `pplx` environment) writes `throughput_local_full_20261107.json`. Without `LT_FULL=1`, the script uses a fixed sample of up to 1,250 test pairs per benchmark, and `report` writes `local_throughput_20261103.json`.
