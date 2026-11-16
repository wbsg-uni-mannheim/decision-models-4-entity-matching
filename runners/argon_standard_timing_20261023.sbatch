#!/bin/bash
# Gemini 4 Argon throughput on the standard tier, 1,000 escalated pairs, 32 workers. Needs Aaron's approval file.
#SBATCH --job-name=em-argon-std-timing
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --output=/path/to/decision-em/logs/argon-std-timing-%j.log
set -euo pipefail
cd /path/to/decision-em
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python -u artifacts/operations/argon_standard_timing_20261023.py --root /path/to/decision-em
