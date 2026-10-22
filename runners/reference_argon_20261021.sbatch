#!/bin/bash
# Reference run on the remaining test pairs: argon. Needs Aaron's approval file. Fill/resume: pass --fill (sol) or --resume (argon).
#SBATCH --job-name=em-ref-argon
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=/ceph/aasteine/decision-em-20261013/logs/reference-argon-%j.log
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python -u artifacts/operations/run_reference_argon_20261021.py --root /ceph/aasteine/decision-em-20261013 --action execute --models argon4 "$@"
