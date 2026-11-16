"""Selective risk of the cascade per benchmark, t from the Sol 2,000 set. Read-only, no inference (Aaron, relayed 6 Nov).

Jev 1.13, PPLX v1.2, Clef, Luna Dec. with B0 at the target risks 0.25, 0.5 and 1 %, t = valsel_eval.tune_risk on the pooled
Sol 2,000 set (cascade_sol2000_runs123_20261105_v2.json), test Runs 1-3. A pair is kept when min(p, 1-p) <= t; the risk is the
share of kept pairs where p >= 0.5 disagrees with the gold label. The target is set on pooled pairs against Sol labels, so
per-benchmark misses are descriptive. Per model, target and benchmark: kept pairs, kept share and risk per run, mean and
sample SD over the runs, runs above the target. Per model and target: the same pooled, runs_above_target over all
benchmark-run cells (of 24) and benchmarks above the target on the 3-run mean (of 8). Pooled risk checked against the v2 file.

    .venv/bin/python src/cascade_selective_risk_benchmark_eval.py --out artifacts/results/cascade_selective_risk_benchmark_20261106.json
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import cascade_solset_runs123_eval as cs
import protocol
import valsel_eval
from thresholds_runs123_eval import test_path as b0_path

B, RUNS, RES = cs.B, cs.RUNS, cs.RES
ROOT = cs.ROOT


def summary(rows, x):
    risks = [v['risk'] for v in rows.values() if v['risk'] is not None]
    return {'per_run': rows, 'kept_share': cs.ms(v['kept_share'] for v in rows.values()),
            'risk': cs.ms(risks) if risks else None, 'runs_above_target': sum(r > x for r in risks),
            'runs_without_kept_pairs': len(rows)-len(risks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    ref = json.loads((RES/'cascade_sol2000_runs123_20261105_v2.json').read_text())
    cs.TUNING[0] = 'sol2000'
    ids, y = {}, {}
    for b in B:
        cases = json.loads((ROOT/'data'/b/'test.json').read_text())
        ids[b] = [c['pair_id'] for c in cases]
        y[b] = np.array([c['label'] for c in cases], dtype=np.int8)
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'tuning_source': 'sol2000',
           'units': 'risk and kept share as fractions', 'models': {}}
    for m in cs.DM4:
        val = cs.tuning_pairs(m)
        tests = {r: cs.test_rows(lambda b: b0_path(m, b, r), ids) for r in RUNS}
        out['models'][m] = {}
        for x in cs.TARGETS:
            t = valsel_eval.tune_risk(val, x)
            per = {b: {} for b in B}
            pooled = {}
            for r in RUNS:
                kept_all = wrong_all = 0
                for b in B:
                    p = tests[r][b]['p']
                    keep = np.minimum(p, 1-p) <= t
                    wrong = int((((p >= .5).astype(np.int8) != y[b]) & keep).sum())
                    kept = int(keep.sum())
                    per[b][str(r)] = {'kept': kept, 'kept_share': kept/len(p), 'wrong': wrong, 'risk': wrong/kept if kept else None}
                    kept_all += kept; wrong_all += wrong
                n = sum(len(v) for v in ids.values())
                pooled[str(r)] = {'kept': kept_all, 'kept_share': kept_all/n, 'wrong': wrong_all, 'risk': wrong_all/kept_all}
                assert abs(pooled[str(r)]['risk']-ref['targets'][m][str(x)]['per_run'][str(r)]['selective_risk']) < 1e-12, (m, x, r)
            bench = {b: summary(per[b], x) for b in B}
            out['models'][m][str(x)] = {
                't': t, 'pooled': summary(pooled, x), 'per_benchmark': bench,
                'cells_above_target': sum(v['runs_above_target'] for v in bench.values()), 'cells': 3*len(B),
                'benchmarks_above_target_on_mean': [b for b in B if bench[b]['risk'] and bench[b]['risk']['mean'] > x]}
            o = out['models'][m][str(x)]
            print(m, x, round(t, 4), 'pooled', round(100*o['pooled']['risk']['mean'], 3), 'kept', round(100*o['pooled']['kept_share']['mean'], 1), 'cells above',
                  o['cells_above_target'], 'bench above', o['benchmarks_above_target_on_mean'],
                  {b: round(100*bench[b]['risk']['mean'], 2) for b in B if bench[b]['risk']}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
