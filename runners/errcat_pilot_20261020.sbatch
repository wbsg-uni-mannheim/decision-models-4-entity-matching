#!/bin/bash
# Error categorization pilot: 100 held-out cases, GPT-6.1 Sol flex, frozen 8-group codebook.
#SBATCH --job-name=em-errcat-pilot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --output=/ceph/aasteine/decision-em-20261013/logs/errcat-pilot-%j.log
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python -u artifacts/operations/error_categorize_live_20261020.py --root /ceph/aasteine/decision-em-20261013 --action run \
  --cases artifacts/astra_error_analysis/errcat_pilot100_cases.json --out runs/error_categorization/sol_flex_pilot100_20261020
