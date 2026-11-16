#!/bin/bash
# Local throughput on the full test sets (30,225 pairs), three runs, table settings, no API calls (7 Nov 2026)
#SBATCH --job-name=ltf-qwen
#SBATCH --partition=gpu-vram-94gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/lt_envs/logs/ltf-qwen_%j.log
E=/path/to/lt_envs
export HF_HUB_CACHE=/path/to/hf_cache/hub HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME=$E/cache TRITON_CACHE_DIR=$E/cache/triton TORCHINDUCTOR_CACHE_DIR=$E/cache/inductor VLLM_CACHE_ROOT=$E/cache/vllm
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export LT_FULL=1 LT_OUT=/path/to/decision-em/runs/local_throughput_full_20261107
cd /path/to/decision-em/runs/local_throughput_20261103
nvidia-smi; lscpu | grep "Model name"
export TOKENIZERS_PARALLELISM=false
for k in 1 2 3; do
  LT_RUN=r$k LT_SEED=$((k-1)) $E/vllm/bin/python local_throughput_20261103.py qwen --sampling api || echo FAILED r$k
done
echo JOB_DONE
