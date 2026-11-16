#!/bin/bash
# Claude Opus 5.5, normal OpenRouter requests on 1,000 remaining test pairs. Needs Aaron's approval file. Fill: pass --fill.
#SBATCH --job-name=em-ref-opus-live
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --output=/path/to/decision-em/logs/reference-opus-live-%j.log
set -euo pipefail
cd /path/to/decision-em
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python -u artifacts/operations/reference_opus_live_sample_20261022.py --root /path/to/decision-em --action run "$@"
