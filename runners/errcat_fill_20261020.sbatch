#!/bin/bash
# Fill pass: re-request cases without a valid result in runs 1-3 (flex, patient retries).
#SBATCH --job-name=em-errcat-fill
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --output=/ceph/aasteine/decision-em-20261013/logs/errcat-fill-%j.log
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1
python3 -c "import json; d=json.load(open('artifacts/astra_error_analysis/preparation_v1/cases_private.json'))['evaluation']; assert len(d)==498; json.dump(d, open('/tmp/errcat_eval498_$SLURM_JOB_ID.json','w'), ensure_ascii=False)"
for r in 1 2 3; do
  .venv/bin/python -u artifacts/operations/error_categorize_live_20261020.py --root /ceph/aasteine/decision-em-20261013 --action run --fill \
    --cases /tmp/errcat_eval498_$SLURM_JOB_ID.json --out runs/error_categorization/sol_flex_full498_run$r \
    --approval artifacts/operations/approval_errcat_full_aaron_20261020.json
done
