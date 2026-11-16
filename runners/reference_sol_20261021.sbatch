#!/bin/bash
# Reference run on the remaining test pairs: sol. Needs Aaron's approval file. Fill/resume: pass --fill (sol) or --resume (argon).
#SBATCH --job-name=em-ref-sol
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=/path/to/decision-em/logs/reference-sol-%j.log
set -euo pipefail
cd /path/to/decision-em
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python -u artifacts/operations/reference_sol_direct_20261021.py --root /path/to/decision-em --action run "$@"
