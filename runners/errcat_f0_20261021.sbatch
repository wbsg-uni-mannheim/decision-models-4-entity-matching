#!/bin/bash
# Error categorization on the F0 error sample (479 cases) x 3 runs, GPT-6.1 Sol flex, frozen 8-group codebook.
# Fill pass afterwards: rerun with FILL=--fill.
#SBATCH --job-name=em-errcat-f0
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --output=/ceph/aasteine/decision-em-20261013/logs/errcat-f0-%j.log
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1
python3 -c "import json; d=json.load(open('artifacts/astra_error_analysis/f0_v1/cases_private.json'))['evaluation']; assert len(d)==479; json.dump(d, open('/tmp/errcat_f0_$SLURM_JOB_ID.json','w'), ensure_ascii=False)"
for r in 1 2 3; do
  .venv/bin/python -u artifacts/operations/error_categorize_live_20261020.py --root /ceph/aasteine/decision-em-20261013 --action run ${FILL:-} \
    --cases /tmp/errcat_f0_$SLURM_JOB_ID.json --out runs/error_categorization/sol_flex_f0_run$r \
    --approval artifacts/operations/approval_errcat_f0_aaron_20261021.json
done
