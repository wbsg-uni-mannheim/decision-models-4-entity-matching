"""Two paired tests for the cascade chapter, t tuned on the Sol-labelled 2,000 set. Read-only, no inference (paper writer, 5 Nov).

Cascades exactly as cascade_solset_runs123_eval.py --tuning sol2000 (cascade_sol2000_runs123_20261105_v2.json): Jev 1.13,
PPLX v1.2, Clef, Luna Dec. with B0, t = valsel_eval.tune_risk on the pooled Sol 2,000 set, decision-model run r with
partner run r (r = 1, 2, 3).
(1) partner_alone, at 0.5 %: + ensemble against the majority of Sol, Opus and Argon alone on all test pairs, + Sol against
    Sol alone; diff = cascade minus partner alone (F1 points, mean over runs); cost_ratio = mean list cost of the partner
    alone on all test pairs / mean list cost of the cascade (decision model on all pairs + partner on the deferred pairs).
(2) target_vs_05: the cascade at 0.25 % and at 1 % minus the cascade at 0.5 %, same model and partner; usd_diff = cost at
    the other risk minus cost at 0.5 % (mean over runs).
Significance: fixed_ci95 = paired bootstrap on the run mean (10,000 replicates, seed 0, pairs drawn per benchmark in
protocol order, the same pairs for both arms and every run); multi_ci95 = multi-bootstrap of multibootstrap_eval.py with
runs drawn jointly for both arms (run r of both arms uses partner run r). Checked against the v2 file and against
majority_alone of cascade_extras_sol2000_runs123_20261105.json.

    .venv/bin/python src/cascade_sol2000_tests_eval.py --out artifacts/results/cascade_sol2000_tests_20261105.json
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


def sig(e):
    return {'fixed_ci95': e['fixed_ci95'], 'multi_ci95': e['multi_ci95'],
            'fixed_significant': e['fixed_ci95'][0] > 0 or e['fixed_ci95'][1] < 0, 'multi_significant': e['multi_significant'],
            'pairing': e['pairing']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    ref = json.loads((RES/'cascade_sol2000_runs123_20261105_v2.json').read_text())
    ext = json.loads((RES/'cascade_extras_sol2000_runs123_20261105.json').read_text())
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
                D.add(('alone', c, r), b, partner[r][c][b][0])
    res = {}
    for m in cs.DM4:
        val = cs.tuning_pairs(m)
        tests = {r: cs.test_rows(lambda b: b0_path(m, b, r), ids) for r in RUNS}
        for x in cs.TARGETS:
            t = valsel_eval.tune_risk(val, x)
            assert abs(t-ref['targets'][m][str(x)]['t']) < 1e-12, (m, x)
            for r in RUNS:
                res[(m, x, r)], preds = cs.cascade(y, tests[r], partner[r], t)
                for c in CONFIGS:
                    for b in B:
                        D.add(('casc', m, x, c, r), b, preds[c][b])
            for c in CONFIGS:
                got = statistics.mean(100*res[(m, x, r)]['configs'][c]['mean_f1'] for r in RUNS)
                assert abs(got-ref['targets'][m][str(x)]['over_runs']['configs'][c]['mean_f1']['mean']) < 1e-9, (m, x, c)
    D.compute()
    for c in CONFIGS:
        got = statistics.mean(100*statistics.mean(D.point[(('alone', c, r), b)] for b in B) for r in RUNS)
        if c == 'majority':
            assert abs(got-ext['majority_alone']['over_runs']['mean_f1']['mean']) < 1e-9, got
    draws = RunDraws()
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': 10_000, 'tuning_source': 'sol2000',
           'units': 'F1 and differences in points; USD at list prices', 'partner_alone': [], 'target_vs_05': []}
    for m in cs.DM4:
        x = cs.X
        for c in CONFIGS:
            spec = {b: ((('dm', m), [('casc', m, x, c, r) for r in RUNS]), (('dm', m), [('alone', c, r) for r in RUNS])) for b in B}
            e = compare(D, draws, spec, None)
            f_c = statistics.mean(100*res[(m, x, r)]['configs'][c]['mean_f1'] for r in RUNS)
            f_a = statistics.mean(100*statistics.mean(D.point[(('alone', c, r), b)] for b in B) for r in RUNS)
            u_c = statistics.mean(res[(m, x, r)]['configs'][c]['usd_list'] for r in RUNS)
            u_a = statistics.mean(alone_usd[(c, r)] for r in RUNS)
            assert abs(e['diff']-(f_c-f_a)) < 1e-9, (m, c)
            out['partner_alone'].append({'model': m, 'target_risk': x, 'config': c, 'partner': 'majority of Sol, Opus, Argon' if c == 'majority' else 'Sol',
                                         'mean_f1_cascade': f_c, 'mean_f1_partner_alone': f_a, 'diff': e['diff'], **sig(e),
                                         'usd_cascade': u_c, 'usd_partner_alone': u_a, 'cost_ratio': u_a/u_c,
                                         'deferred_share': statistics.mean(res[(m, x, r)]['deferred_share'] for r in RUNS)})
        for xo in (.0025, .01):
            for c in CONFIGS:
                spec = {b: ((('dm', m), [('casc', m, xo, c, r) for r in RUNS]), (('dm', m), [('casc', m, x, c, r) for r in RUNS])) for b in B}
                e = compare(D, draws, spec, None)
                f = {z: statistics.mean(100*res[(m, z, r)]['configs'][c]['mean_f1'] for r in RUNS) for z in (xo, x)}
                u = {z: statistics.mean(res[(m, z, r)]['configs'][c]['usd_list'] for r in RUNS) for z in (xo, x)}
                dsh = {z: statistics.mean(res[(m, z, r)]['deferred_share'] for r in RUNS) for z in (xo, x)}
                assert abs(e['diff']-(f[xo]-f[x])) < 1e-9, (m, xo, c)
                out['target_vs_05'].append({'model': m, 'target_risk': xo, 'reference_risk': x, 'config': c,
                                            'mean_f1': f[xo], 'mean_f1_05': f[x], 'diff': e['diff'], **sig(e),
                                            'usd': u[xo], 'usd_05': u[x], 'usd_diff': u[xo]-u[x],
                                            'deferred_share': dsh[xo], 'deferred_share_05': dsh[x]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    for k in ('partner_alone', 'target_vs_05'):
        for e in out[k]:
            print(k, e['model'], e['target_risk'], e['config'], round(e['diff'], 2), [round(v, 2) for v in e['fixed_ci95']],
                  [round(v, 2) for v in e['multi_ci95']], round(e.get('cost_ratio', e.get('usd_diff')), 2), flush=True)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
