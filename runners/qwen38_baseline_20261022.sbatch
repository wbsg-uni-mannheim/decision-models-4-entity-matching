#!/bin/bash
# Qwen3.8-27B (deepinfra bf16), generative B0, full test sets, runs 1-3 (sequential). Resume: add --resume.
#SBATCH --job-name=em-qwen38-b0
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=/path/to/decision-em/logs/qwen38-b0-%j.log
set -euo pipefail
cd /path/to/decision-em
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1
for r in ${RUNS:-1 2 3}; do
  .venv/bin/python -u artifacts/operations/run_qwen38_baseline_20261022.py --root /path/to/decision-em --action execute --run-id $r "$@"
done
