"""Supplement to cascade_fixed_t_20261107.json for the tables with the fixed threshold t = 0.05 (Aaron, relayed 7 Nov:
chapter 5 reports the cascades without labeled pairs at t = 0.05, chapter 6 the cascades tuned on the Sol 2,000 set).
Read-only, no inference. Same runs, partners, cascade rule and pairing as cascade_fixed_t_eval.py.

alone: per model F1 per benchmark and mean F1 (mean and sample SD over the runs) and list cost of the model alone.
deferred: per model at t = 0.05, share of the model's errors in the deferred pairs and the accuracy of the ensemble majority,
  Sol and the model on the deferred pairs, per run and over the runs (as cascade_extras_runs123_eval.py).
tuned_vs_fixed: per model and partner, tuned cascade (Sol 2,000, 0.5 %) minus the t = 0.05 cascade (F1 points, intervals of
  cascade_fixed_t_20261107.json with the sign flipped), cost ratio and deferred shares.
literature: PPLX v1.2 with B0 and with RULESFS10_2000 (run-1 rules, 3 test runs), alone and with the ensemble (and Sol) at
  t = 0.05: F1 per benchmark per run, mean and SD, how often above both published values (sources/literature.json, as
  later_tables_runs123_eval.py), and the paired bootstrap on the run mean plus multi-bootstrap of RULESFS10_2000 against B0
  (alone and cascade; independent run draws, separate API calls).

    .venv/bin/python src/cascade_fixed_t_supplement_eval.py --paper-sources <dir with literature.json> --out artifacts/results/cascade_fixed_t_supplement_20261107.json
"""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import cascade_b0_valsel_eval as cv
import cascade_solset_runs123_eval as cs
from multibootstrap_eval import Data as BootData, RunDraws, compare
from thresholds_runs123_eval import test_path as b0_path

B, RUNS, RES = cs.B, cs.RUNS, cs.RES
T = .05
CONFIGS = ('sol61', 'majority')


def flip(s):
    return {'diff': -s['diff'], 'fixed_ci95': [-s['fixed_ci95'][1], -s['fixed_ci95'][0]], 'multi_ci95': [-s['multi_ci95'][1], -s['multi_ci95'][0]],
            'fixed_significant': s['fixed_significant'], 'multi_significant': s['multi_significant'], 'pairing': s['pairing']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper-sources', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    fx = json.loads((RES/'cascade_fixed_t_20261107.json').read_text())
    cells = {(x['model'], x['partner']): x for x in fx['cells'] if x['t'] == T}
    lit = json.loads((args.paper_sources/'literature.json').read_text())['benchmarks']
    published = {b: max(round(lit[b]['trained']['f1'], 2), round(lit[b]['zero_shot']['f1'], 2)) for b in B}
    D = BootData()
    ids, y = D.ids, D.y
    dec1 = cv.ensemble()
    partner = {r: cs.partner_run(r, ids, dec1)[0] for r in RUNS}
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 't': T, 'replicates': 10_000,
           'units': 'F1 and differences in points (per_run F1 as fractions); shares as fractions; USD at list prices',
           'alone': {}, 'deferred': {}, 'tuned_vs_fixed': [], 'literature': {}}
    for m in cs.DM4:
        pr, dres = {}, {}
        for r in RUNS:
            test = cs.test_rows(lambda b: b0_path(m, b, r), ids)
            res, preds = cs.cascade(y, test, partner[r], T)
            assert res['deferred'] == cells[(m, 'sol61')]['per_run'][str(r)]['deferred'], (m, r)
            pr[str(r)] = res
            err = err_d = n_d = 0
            correct = {'majority': 0, 'sol61': 0, 'alone': 0}
            for b in B:
                p = test[b]['p']
                alone = (p >= .5).astype('int8')
                d = np.minimum(p, 1-p) > T
                wrong = alone != y[b]
                err += int(wrong.sum()); err_d += int(wrong[d].sum()); n_d += int(d.sum())
                correct['alone'] += int((~wrong[d]).sum())
                for c in CONFIGS:
                    correct[c] += int((partner[r][c][b][0][d] == y[b][d]).sum())
            assert n_d == res['deferred'], (m, r)
            dres[str(r)] = {'deferred': n_d, 'deferred_share': n_d/sum(len(v) for v in ids.values()), 'errors_alone': err,
                            'error_share_in_deferred': err_d/err, 'accuracy_on_deferred': {c: v/n_d for c, v in correct.items()}}
        o = cs.over_runs(pr)
        out['alone'][m] = {'mean_f1': o['configs']['alone']['mean_f1'], 'per_benchmark': o['configs']['alone']['per_benchmark'],
                           'usd_list': o['configs']['alone']['usd_list'], 'errors': o['configs']['alone']['errors']}
        rs = list(dres.values())
        out['deferred'][m] = {'per_run': dres, 'over_runs': {
            'deferred_share': cs.ms(x['deferred_share'] for x in rs), 'error_share_in_deferred': cs.ms(x['error_share_in_deferred'] for x in rs),
            'accuracy_on_deferred': {c: cs.ms(x['accuracy_on_deferred'][c] for x in rs) for c in ('majority', 'sol61', 'alone')}}}
        for c in CONFIGS:
            x = cells[(m, c)]
            assert abs(o['configs'][c]['mean_f1']['mean']-x['mean_f1']['mean']) < 1e-9, (m, c)
            out['tuned_vs_fixed'].append({'model': m, 'partner': c, 't_tuned': x['tuned']['t'], 'mean_f1_tuned': x['tuned']['mean_f1'], 'mean_f1_fixed': x['mean_f1']['mean'],
                                          'tuned_minus_fixed': flip(x['vs_tuned']), 'usd_tuned': x['tuned']['usd_list'], 'usd_fixed': x['usd_list']['mean'],
                                          'cost_ratio_fixed_over_tuned': x['usd_list']['mean']/x['tuned']['usd_list'],
                                          'deferred_share_tuned': x['tuned']['deferred_share'], 'deferred_share_fixed': x['deferred_share']['mean']})
        print(m, 'alone', round(o['configs']['alone']['mean_f1']['mean'], 2), 'error share in deferred', round(out['deferred'][m]['over_runs']['error_share_in_deferred']['mean'], 3), flush=True)

    # literature: PPLX v1.2 B0 and RULESFS10_2000 at t = 0.05
    series = {'b0': lambda r: (lambda b: b0_path('pplx_v12', b, r)),
              'lab': lambda r: (lambda b: cs.ROOT/cs.SERIES['RULESFS10_fixed'][1][r]/'pplx_v12/RULESFS10_2000'/b/'test.jsonl')}
    runs = {}
    for s, path_of in series.items():
        runs[s] = {}
        for r in RUNS:
            res, preds = cs.cascade(y, cs.test_rows(path_of(r), ids), partner[r], T)
            runs[s][str(r)] = res
            for c in ('alone',)+CONFIGS:
                for b in B:
                    D.add((s, c, r), b, preds[c][b])
    D.compute()
    draws = RunDraws()
    groups = {'b0': ('dm', 'pplx_v12'), 'lab': ('lab', 'RULESFS10_fixed', 'pplx_v12', '2000')}
    for s in series:
        o = cs.over_runs(runs[s])
        for c in ('alone',)+CONFIGS:
            f1 = {b: {r: runs[s][r]['configs'][c]['per_benchmark'][b] for r in runs[s]} for b in B}
            out['literature'][f'{s}_{c}'] = {
                'f1': f1, 'per_benchmark': o['configs'][c]['per_benchmark'], 'mean_f1': o['configs'][c]['mean_f1'], 'usd_list': o['configs'][c]['usd_list'],
                'deferred_share': o['deferred_share'],
                'above_both_published_per_run': {r: sum(round(100*f1[b][r], 2) > published[b] for b in B) for r in runs[s]},
                'above_both_published_on_mean': sum(round(o['configs'][c]['per_benchmark'][b]['mean'], 2) > published[b] for b in B)}
    x = out['literature']['b0_majority']['mean_f1']['mean']
    assert abs(x-cells[('pplx_v12', 'majority')]['mean_f1']['mean']) < 1e-9, x
    out['literature']['published'] = published
    out['literature']['significance'] = []
    for c in ('alone',)+CONFIGS:
        spec = {b: ((groups['lab'], [('lab', c, r) for r in RUNS]), (groups['b0'], [('b0', c, r) for r in RUNS])) for b in B}
        e = compare(D, draws, spec, None)
        out['literature']['significance'].append({'comparison': f'RULESFS10_2000 {c} minus B0 {c} (t = 0.05)', 'diff': e['diff'], 'fixed_ci95': e['fixed_ci95'],
                                                  'multi_ci95': e['multi_ci95'], 'fixed_significant': e['fixed_ci95'][0] > 0 or e['fixed_ci95'][1] < 0,
                                                  'multi_significant': e['multi_significant'], 'pairing': e['pairing']})
    for k, v in out['literature'].items():
        if isinstance(v, dict) and 'mean_f1' in v:
            print(k, round(v['mean_f1']['mean'], 2), 'above both', v['above_both_published_on_mean'], v['above_both_published_per_run'], flush=True)
    for e in out['literature']['significance']:
        print(e['comparison'], round(e['diff'], 2), [round(v, 2) for v in e['fixed_ci95']], [round(v, 2) for v in e['multi_ci95']], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1, default=lambda o: o.item())
    print('wrote', args.out)


if __name__ == '__main__':
    main()
