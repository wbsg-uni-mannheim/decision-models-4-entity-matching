#!/bin/bash
# Local throughput study (3 Nov 2026): download open weights into the shared HF cache. No GPU, no API calls.
#SBATCH --job-name=lt-download
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/decision-em/runs/local_throughput_20261103/logs/download_%j.log
export HF_HUB_CACHE=/path/to/hf_cache/hub
export HF_HUB_DISABLE_TELEMETRY=1 HF_XET_CACHE=/path/to/hf_cache/xet
PY=/usr/bin/python3
for i in $(seq 1 120); do grep -q PPLX_DONE /path/to/lt_envs/setup.log && break; sleep 30; done
/path/to/lt_envs/pplx/bin/hf download perplexity-ai/pplx-decider-v1.2
/path/to/lt_envs/pplx/bin/hf download Qwen/Qwen3.8-27B
/path/to/lt_envs/pplx/bin/hf download roberta-base
du -sh $HF_HUB_CACHE/models--perplexity-ai--pplx-decider-v1.2 $HF_HUB_CACHE/models--Qwen--Qwen3.8-27B
echo DOWNLOAD_DONE
