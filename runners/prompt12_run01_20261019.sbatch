#!/bin/bash
# Paid prompt12 run (11 prompts x 7 models x 30,225 pairs). Requires Aaron's approval file,
# the frozen manifest and an authorized ledger cap. Submit held (--hold) and release deliberately.
# Threads mostly wait on HTTP; 8 CPUs cover 7 models x workers threads plus JSON parsing.
#SBATCH --job-name=em-prompt12-run01
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/ceph/aasteine/decision-em-20261013/logs/prompt12-run01-%j.log
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
exec .venv/bin/python -u artifacts/operations/run_prompt12_20261019.py --root /ceph/aasteine/decision-em-20261013 --action execute
