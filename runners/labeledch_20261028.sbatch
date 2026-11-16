#!/bin/bash
# Labelled-data chapter, Jev/PPLX runs. Usage: sbatch labeledch_20261028.sbatch --run 1 --group rules|fewshot|fewshot2|rulesfs [--resume]
# Needs: frozen manifest for the group (--action freeze), Aaron s approval file for the group, authorized ledger.
# Graceful stop: scancel --signal=TERM --batch <job>.
#SBATCH --job-name=em-labeledch
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/path/to/decision-em/logs/labeledch-%j.log
set -euo pipefail
cd /path/to/decision-em
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
exec .venv/bin/python -u artifacts/operations/run_labeled_chapter_20261028.py --root /path/to/decision-em --action execute "$@"
