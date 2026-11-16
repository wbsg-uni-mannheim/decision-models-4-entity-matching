#!/bin/bash
# Reference run on the remaining test pairs: argon. Needs Aaron's approval file. Fill/resume: pass --fill (sol) or --resume (argon).
#SBATCH --job-name=em-sota-argon
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=/path/to/decision-em/logs/sota-argon-%j.log
set -euo pipefail
cd /path/to/decision-em
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python -u artifacts/operations/run_sota_argon_runs23_20261103.py --root /path/to/decision-em --action execute --models argon4 "$@"
