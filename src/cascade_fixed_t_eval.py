"""Cascades with a fixed threshold, no labeled pairs for tuning (Chris, relayed 7 Nov). Read-only, no inference.

Jev 1.13, PPLX v1.2, Clef, Luna Dec. with B0; partners Sol and the majority of Sol, Opus, Argon; decision-model run r with
partner run r. Fixed t in FIXED: the model keeps a pair when min(p, 1-p) <= t (t = 0.1: p <= 0.1 or p >= 0.9) and defers
the rest; kept pairs are a match when p >= 0.5 (cascade_solset_runs123_eval.cascade). Per cell: F1 per benchmark and mean
F1 per run, mean and sample SD over the runs; deferred share; list cost; selective risk against gold on the kept pairs.
Comparisons (diff = fixed-t cascade minus other arm, F1 points): against the model alone, against the tuned cascade (t from the
Sol 2,000 set at 0.5 %, cascade_sol2000_runs123_20261105_v2.json) and against the partner alone on all pairs.
fixed_ci95: paired bootstrap on the run mean (10,000 replicates, seed 0, pairs drawn per benchmark in protocol order);
multi_ci95: runs drawn with replacement (rng seed 1), jointly for both arms because run r of both uses partner run r.
grid: mean F1, deferred share and cost for t = 0.01 ... 0.49 (no bootstrap) and the grid t closest to the tuned cascade in cost
and in F1.

    .venv/bin/python src/cascade_fixed_t_eval.py --out artifacts/results/cascade_fixed_t_20261107.json
"""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import cascade_b0_valsel_eval as cv
import cascade_solset_runs123_eval as cs
import valsel_eval
from multibootstrap_eval import Data as BootData, RunDraws, compare
from thresholds_runs123_eval import test_path as b0_path

B, RUNS, RES = cs.B, cs.RUNS, cs.RES
CONFIGS = ('sol61', 'majority')
FIXED = (.05, .1, .2, .3)
GRID = [round(x/100, 2) for x in range(1, 50)]


def sig(e):
    return {'diff': e['diff'], 'fixed_ci95': e['fixed_ci95'], 'multi_ci95': e['multi_ci95'],
            'fixed_significant': e['fixed_ci95'][0] > 0 or e['fixed_ci95'][1] < 0, 'multi_significant': e['multi_significant'], 'pairing': e['pairing']}


def mean_over(res, c, key):
    return statistics.mean((100*x['configs'][c]['mean_f1'] if key == 'f1' else x['configs'][c]['usd_list'] if key == 'usd' else x[key]) for x in res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    ref = json.loads((RES/'cascade_sol2000_runs123_20261105_v2.json').read_text())
    cs.TUNING[0] = 'sol2000'
    D = BootData()
    ids, y = D.ids, D.y
    dec1 = cv.ensemble()
    partner, alone_usd = {}, {}
    for r in RUNS:
        partner[r] = cs.partner_run(r, ids, dec1)[0]
        for c in CONFIGS:
            alone_usd[(c, r)] = sum(float(partner[r][c][b][1].sum()) for b in B)
            for b in B:
                D.add(('part', c, r), b, partner[r][c][b][0])
    res, grid = {}, {}
    for m in cs.DM4:
        t_tuned = valsel_eval.tune_risk(cs.tuning_pairs(m), cs.X)
        assert abs(t_tuned-ref['table9'][m]['t']) < 1e-12, m
        tests = {r: cs.test_rows(lambda b: b0_path(m, b, r), ids) for r in RUNS}
        for r in RUNS:
            res[(m, 'tuned', r)], preds = cs.cascade(y, tests[r], partner[r], t_tuned)
            for b in B:
                D.add(('alone', m, r), b, preds['alone'][b])
                for c in CONFIGS:
                    D.add(('tuned', m, c, r), b, preds[c][b])
            for t in FIXED:
                res[(m, t, r)], preds = cs.cascade(y, tests[r], partner[r], t)
                for b in B:
                    for c in CONFIGS:
                        D.add(('fix', m, t, c, r), b, preds[c][b])
        for c in CONFIGS:
            got = mean_over([res[(m, 'tuned', r)] for r in RUNS], c, 'f1')
            assert abs(got-ref['table9'][m]['over_runs']['configs'][c]['mean_f1']['mean']) < 1e-9, (m, c)
        g = {}
        for t in GRID:
            rs = [cs.cascade(y, tests[r], partner[r], t)[0] for r in RUNS]
            g[str(t)] = {'deferred_share': mean_over(rs, None, 'deferred_share'), 'selective_risk': mean_over(rs, None, 'selective_risk'),
                         **{c: {'mean_f1': mean_over(rs, c, 'f1'), 'usd_list': mean_over(rs, c, 'usd')} for c in CONFIGS}}
        tuned = [res[(m, 'tuned', r)] for r in RUNS]
        grid[m] = {'t_tuned': t_tuned, 'points': g, 'closest': {c: {
            'in_cost': min(GRID, key=lambda t: abs(g[str(t)][c]['usd_list']-mean_over(tuned, c, 'usd'))),
            'in_f1': min(GRID, key=lambda t: abs(g[str(t)][c]['mean_f1']-mean_over(tuned, c, 'f1'))),
            'in_deferred_share': min(GRID, key=lambda t: abs(g[str(t)]['deferred_share']-mean_over(tuned, None, 'deferred_share')))} for c in CONFIGS}}
        print(m, 'grid done', grid[m]['closest'], flush=True)
    D.compute()
    draws = RunDraws()
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': 10_000, 'fixed_t': FIXED,
           'units': 'per_run F1 as fractions; mean F1, differences and intervals in F1 points; shares as fractions; USD at list prices',
           'cells': [], 'tuned': {}, 'grid': grid}
    for m in cs.DM4:
        tuned = [res[(m, 'tuned', r)] for r in RUNS]
        out['tuned'][m] = {'t': grid[m]['t_tuned'], 'over_runs': cs.over_runs({str(r): res[(m, 'tuned', r)] for r in RUNS})}
        for t in FIXED:
            pr = {str(r): res[(m, t, r)] for r in RUNS}
            for c in CONFIGS:
                def spec(other):
                    return {b: ((('dm', m), [('fix', m, t, c, r) for r in RUNS]), (('dm', m), [(*other, r) for r in RUNS])) for b in B}
                e_alone, e_tuned, e_part = (compare(D, draws, spec(o), None) for o in (('alone', m), ('tuned', m, c), ('part', c)))
                f_fix = mean_over(pr.values(), c, 'f1')
                f_part = statistics.mean(100*statistics.mean(D.point[(('part', c, r), b)] for b in B) for r in RUNS)
                assert abs(e_part['diff']-(f_fix-f_part)) < 1e-9 and abs(e_tuned['diff']-(f_fix-mean_over(tuned, c, 'f1'))) < 1e-9, (m, t, c)
                o = cs.over_runs(pr)
                out['cells'].append({
                    'model': m, 't': t, 'partner': c, 'per_run': {r: {k: v for k, v in x.items() if k != 'configs'} | {'config': x['configs'][c], 'alone': x['configs']['alone']}
                                                                 for r, x in pr.items()},
                    'mean_f1': o['configs'][c]['mean_f1'], 'per_benchmark': o['configs'][c]['per_benchmark'],
                    'deferred_share': o['deferred_share'], 'selective_risk': o['selective_risk'], 'usd_list': o['configs'][c]['usd_list'],
                    'tuned': {'t': grid[m]['t_tuned'], 'mean_f1': mean_over(tuned, c, 'f1'), 'usd_list': mean_over(tuned, c, 'usd'),
                              'deferred_share': mean_over(tuned, None, 'deferred_share'), 'selective_risk': mean_over(tuned, None, 'selective_risk')},
                    'alone_mean_f1': o['configs']['alone']['mean_f1']['mean'], 'partner_alone': {'mean_f1': f_part, 'usd_list': statistics.mean(alone_usd[(c, r)] for r in RUNS)},
                    'vs_alone': sig(e_alone), 'vs_tuned': sig(e_tuned), 'vs_partner_alone': sig(e_part)})
                x = out['cells'][-1]
                print(m, t, c, 'F1', round(x['mean_f1']['mean'], 2), 'def', round(100*x['deferred_share']['mean'], 1), 'risk', round(100*x['selective_risk']['mean'], 2),
                      'usd', round(x['usd_list']['mean'], 2), '| tuned', round(x['tuned']['mean_f1'], 2), round(x['tuned']['usd_list'], 2),
                      '| vs tuned', round(e_tuned['diff'], 2), [round(v, 2) for v in e_tuned['fixed_ci95']], [round(v, 2) for v in e_tuned['multi_ci95']],
                      '| vs partner', round(e_part['diff'], 2), [round(v, 2) for v in e_part['multi_ci95']], '| vs alone', round(e_alone['diff'], 2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1, default=lambda o: o.item())
    print('wrote', args.out)


if __name__ == '__main__':
    main()
