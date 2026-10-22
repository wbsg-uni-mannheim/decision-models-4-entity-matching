#!/bin/bash
# Resume of the prompt12 run with runner v2 (DNS cache, graceful SIGTERM, pplx_v12 with own B0).
# Graceful stop: scancel --signal=TERM --batch <job>  (drains in-flight requests, commits journals).
#SBATCH --job-name=em-prompt12-run01-v2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/ceph/aasteine/decision-em-20261013/logs/prompt12-run01-v2-%j.log
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
exec .venv/bin/python -u artifacts/operations/run_prompt12_20261019.py --root /ceph/aasteine/decision-em-20261013 --action execute --resume
