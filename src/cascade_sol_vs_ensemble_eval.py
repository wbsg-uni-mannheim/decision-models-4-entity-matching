"""Single frontier LLM against the ensemble as cascade partner, t = 0.05. Read-only, no inference (abstract check, relayed 9 Nov).

Jev 1.13, PPLX v1.2, Clef, Luna Dec. with B0, fixed t = 0.05, decision-model run r with partner run r (cascade_fixed_t_20261107.json,
asserted). Per model: X + Sol cascade minus X + ensemble cascade, and X + Sol cascade minus the ensemble majority (Sol, Opus, Argon)
alone on all test pairs. diff in F1 points, mean over the runs and the eight benchmarks. fixed_ci95: paired bootstrap on the run mean
(multibootstrap_eval.Data: 10,000 replicates, seed 0, pairs drawn per benchmark in protocol order, the same pairs for both arms and
every run); multi_ci95: runs drawn with replacement (rng seed 1), jointly for both arms because both use partner run r.
both_include_zero: both intervals contain zero. Cost: list USD, mean over the runs, as Table VII.

    .venv/bin/python src/cascade_sol_vs_ensemble_eval.py --out artifacts/results/cascade_sol_vs_ensemble_20261109.json
"""
import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import cascade_b0_valsel_eval as cv
import cascade_solset_runs123_eval as cs
from multibootstrap_eval import Data as BootData, RunDraws, compare
from thresholds_runs123_eval import test_path as b0_path

B, RUNS, RES = cs.B, cs.RUNS, cs.RES
T = .05
FIXED = RES/'cascade_fixed_t_20261107.json'


def includes_zero(c):
    return c[0] <= 0 <= c[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    fixed = {(x['model'], x['partner']): x for x in json.loads(FIXED.read_text())['cells'] if x['t'] == T}
    D = BootData()
    ids, y = D.ids, D.y
    dec1 = cv.ensemble()
    partner, ens_usd = {}, {}
    for r in RUNS:
        partner[r] = cs.partner_run(r, ids, dec1)[0]
        ens_usd[r] = sum(float(partner[r]['majority'][b][1].sum()) for b in B)
        for b in B:
            D.add(('ens', r), b, partner[r]['majority'][b][0])
    res = {}
    for m in cs.DM4:
        for r in RUNS:
            res[(m, r)], preds = cs.cascade(y, cs.test_rows(lambda b: b0_path(m, b, r), ids), partner[r], T)
            for c in ('sol61', 'majority'):
                for b in B:
                    D.add(('casc', m, c, r), b, preds[c][b])
        for c in ('sol61', 'majority'):
            got = statistics.mean(100*res[(m, r)]['configs'][c]['mean_f1'] for r in RUNS)
            assert abs(got-fixed[(m, c)]['mean_f1']['mean']) < 1e-9, (m, c)
    D.compute()
    draws = RunDraws()

    def cmp(m, other):
        e = compare(D, draws, {b: ((('dm', m), [('casc', m, 'sol61', r) for r in RUNS]), (('dm', m), [other(r) for r in RUNS])) for b in B}, None)
        return {'diff': e['diff'], 'fixed_ci95': e['fixed_ci95'], 'multi_ci95': e['multi_ci95'], 'fixed_includes_zero': includes_zero(e['fixed_ci95']),
                'multi_includes_zero': includes_zero(e['multi_ci95']), 'both_include_zero': includes_zero(e['fixed_ci95']) and includes_zero(e['multi_ci95']), 'pairing': e['pairing']}

    ens_f1 = statistics.mean(100*statistics.mean(D.point[(('ens', r), b)] for b in B) for r in RUNS)
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 't': T, 'replicates': 10_000,
           'input_sha256': {FIXED.name: hashlib.sha256(FIXED.read_bytes()).hexdigest()},
           'units': 'F1, differences and intervals in points; USD at list prices',
           'ensemble_alone': {'mean_f1': ens_f1, 'usd_list': statistics.mean(ens_usd.values())}, 'models': {}}
    for m in cs.DM4:
        f = {c: statistics.mean(100*res[(m, r)]['configs'][c]['mean_f1'] for r in RUNS) for c in ('sol61', 'majority')}
        u = {c: statistics.mean(res[(m, r)]['configs'][c]['usd_list'] for r in RUNS) for c in ('sol61', 'majority')}
        a, z = cmp(m, lambda r: ('casc', m, 'majority', r)), cmp(m, lambda r: ('ens', r))
        out['models'][m] = {'mean_f1_sol_cascade': f['sol61'], 'mean_f1_ensemble_cascade': f['majority'], 'usd_sol_cascade': u['sol61'], 'usd_ensemble_cascade': u['majority'],
                            'cost_ratio_ensemble_cascade_over_sol_cascade': u['majority']/u['sol61'], 'cost_ratio_ensemble_alone_over_sol_cascade': out['ensemble_alone']['usd_list']/u['sol61'],
                            'sol_cascade_minus_ensemble_cascade': a, 'sol_cascade_minus_ensemble_alone': z}
        for k, v in (('vs ens cascade', a), ('vs ens alone', z)):
            print(m, k, round(v['diff'], 2), [round(x, 2) for x in v['fixed_ci95']], [round(x, 2) for x in v['multi_ci95']], 'both include 0:', v['both_include_zero'], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1, default=lambda o: o.item())
    print('wrote', args.out)


if __name__ == '__main__':
    main()
