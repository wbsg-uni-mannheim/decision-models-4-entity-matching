#!/bin/bash
# Local throughput on the full test sets (30,225 pairs), three runs, table settings, no API calls (7 Nov 2026)
#SBATCH --job-name=ltf-glm
#SBATCH --partition=gpu-vram-94gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/lt_envs/logs/ltf-glm_%j.log
E=/path/to/lt_envs
export HF_HUB_CACHE=/path/to/hf_cache/hub HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME=$E/cache TRITON_CACHE_DIR=$E/cache/triton TORCHINDUCTOR_CACHE_DIR=$E/cache/inductor VLLM_CACHE_ROOT=$E/cache/vllm
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export LT_FULL=1 LT_OUT=/path/to/decision-em/runs/local_throughput_full_20261107
cd /path/to/decision-em/runs/local_throughput_20261103
nvidia-smi; lscpu | grep "Model name"
export VLLM_ENGINE_READY_TIMEOUT_S=3600 PYTHONUNBUFFERED=1 VLLM_CONFIG_ROOT=$E/config/vllm CUDA_CACHE_PATH=$E/cache/cuda DG_JIT_CACHE_DIR=$E/cache/deepgemm FLASHINFER_WORKSPACE_BASE=$E/cache/flashinfer TILELANG_CACHE_DIR=$E/cache/tilelang TILELANG_TMP_DIR=$E/cache/tilelang/tmp
export CUDA_HOME=/usr/local/cuda-13.3 PATH=$E/vllm/bin:/usr/local/cuda-13.3/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
for r in r1 r2 r3; do
  LT_RUN=$r LT_EAGER=0 LT_GLM_MAX_SEQS=64 $E/vllm/bin/python local_throughput_20261103.py glm || echo FAILED $r
done
echo JOB_DONE
