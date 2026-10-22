#!/bin/bash
# Error categorization census (all not yet assigned errors of Jev, PPLX v1.2, Sol, Opus, Argon) x 3 runs,
# GPT-6.1 Sol flex, low reasoning, frozen 8-group codebook. Aaron's approval 2026-10-22, total cap 8 USD.
#SBATCH --job-name=em-errcat-census
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --output=/ceph/aasteine/decision-em-20261013/logs/errcat-census-%j.log
set -euo pipefail
cd /ceph/aasteine/decision-em-20261013
export PYTHONPATH=src:artifacts/operations PYTHONDONTWRITEBYTECODE=1
OPS=artifacts/operations
CASES=/tmp/errcat_census_$SLURM_JOB_ID.json
python3 -c "import json; d=json.load(open('artifacts/astra_error_analysis/census_v1/cases_private.json'))['evaluation']; json.dump(d, open('$CASES','w'), ensure_ascii=False); print(len(d), 'cases')"
spent() { python3 -c "import json,glob; print(sum(json.load(open(p))['cost_usd'] for p in glob.glob('runs/error_categorization/sol_flex_census_run*/summary*.json')))"; }
for r in 1 2 3; do
  .venv/bin/python -u $OPS/error_categorize_live_20261020.py --root /ceph/aasteine/decision-em-20261013 --action run \
    --cases $CASES --out runs/error_categorization/sol_flex_census_run$r --approval $OPS/approval_errcat_census_aaron_20261022.json
done
for r in 1 2 3; do
  missing=$(python3 -c "import json,glob; print(json.load(open(sorted(glob.glob('runs/error_categorization/sol_flex_census_run$r/summary*.json'))[-1]))['missing_after_this_pass'])")
  total=$(spent)
  echo "run $r missing $missing, census spent so far $total USD"
  if [ "$missing" -gt 0 ] && python3 -c "import sys; sys.exit(0 if $total + 0.5 <= 8.0 else 1)"; then
    .venv/bin/python -u $OPS/error_categorize_live_20261020.py --root /ceph/aasteine/decision-em-20261013 --action run --fill \
      --cases $CASES --out runs/error_categorization/sol_flex_census_run$r --approval $OPS/approval_errcat_census_fill_aaron_20261022.json
  fi
done
echo "census total $(spent) USD"
