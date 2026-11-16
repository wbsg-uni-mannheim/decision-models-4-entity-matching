#!/bin/bash
# Sol-labelled 200/4000 sets (labelsets_sol_20261028.py). Paid: needs Aaron's approval file and authorize-budget.
# Usage: sbatch labelsets_sol_20261028.sbatch [--stages known topup al semi export] [--benchmarks ...]
# Resumable: labels.jsonl per benchmark is a cache; a rerun asks only missing pairs.
# Killed job: --action close-stale --commitment labelsets-sol-<job>-<t> reconciles from journal/calls.jsonl.
#SBATCH --job-name=em-labelsets-sol
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/path/to/decision-em/logs/labelsets-sol-%j.log
set -euo pipefail
cd /path/to/decision-em
export PYTHONDONTWRITEBYTECODE=1
exec /path/to/envs/ditto-modern/bin/python -u artifacts/operations/labelsets_sol_20261028.py --action label "$@"
