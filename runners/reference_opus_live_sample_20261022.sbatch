#!/bin/bash
# Claude Opus 5.5, normal OpenRouter requests on 1,000 remaining test pairs. Needs Aaron's approval file. Fill: pass --fill.
#SBATCH --job-name=em-ref-opus-live
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --output=/ceph/aasteine/decision-em-20261013/logs/reference-opus-live-%j.log
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python -u artifacts/operations/reference_opus_live_sample_20261022.py --root /ceph/aasteine/decision-em-20261013 --action run "$@"
