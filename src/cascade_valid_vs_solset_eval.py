"""Cascade with t from the official validation sets against the same cascade with t from the Sol-labelled 2,000 / 200 set.
Read-only, no inference (paper writer, 5 Nov).

Cells: Jev 1.13, PPLX v1.2, Clef, Luna Dec. with B0 at the target risks 0.25, 0.5 and 1 %, + Sol and + ensemble majority;
also the RULESFS10_2000 cascade of PPLX v1.2 at 0.5 % (Table XI; Sol t from its outputs on the Sol 2,000 validation split,
the same for both Sol sets). t and cascades exactly as cascade_solset_runs123_eval.py (decision-model run r with partner
run r; t tuned once and applied to every run). Per cell: diff = mean over the three runs of the mean F1 with valid t
minus with Sol t (points); fixed_ci95: paired bootstrap on the run mean (10,000 replicates, seed 0, pairs drawn per
benchmark in protocol order, the same pairs for both arms and every run); multi_ci95: multi-bootstrap with runs drawn
jointly for both arms (same decision-model and partner run r); cost difference (list USD, valid minus Sol, mean over runs)
and deferred-share difference. Checked against later_tables_{valid,sol200,sol2000}_runs123_20261105[_v2].json.

    .venv/bin/python src/cascade_valid_vs_solset_eval.py --out artifacts/results/cascade_valid_vs_solset_20261105.json
"""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import cascade_b0_valsel_eval as cv
import cascade_solset_runs123_eval as cs
import labeled_valsel_cascade_eval as lv
import valsel_eval
from multibootstrap_eval import Data as BootData, RunDraws, compare
from thresholds_runs123_eval import test_path as b0_path

B, RUNS, RES = cs.B, cs.RUNS, cs.RES
SOURCES = ('valid', 'sol200', 'sol2000')
CONFIGS = ('sol61', 'majority')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    ref = {'valid': json.loads((RES/'later_tables_valid_runs123_20261105.json').read_text()),
           **{s: json.loads((RES/f'later_tables_{s}_runs123_20261105_v2.json').read_text()) for s in ('sol200', 'sol2000')}}
    D = BootData()
    ids, y = D.ids, D.y
    dec1 = cv.ensemble()
    partner = {r: cs.partner_run(r, ids, dec1)[0] for r in RUNS}
    cells, res = {}, {}
    for m in cs.DM4:
        tests = {r: cs.test_rows(lambda b: b0_path(m, b, r), ids) for r in RUNS}
        for s in SOURCES:
            cs.TUNING[0] = s
            val = cs.tuning_pairs(m)
            for x in cs.TARGETS:
                t = valsel_eval.tune_risk(val, x)
                for r in RUNS:
                    out, preds = cs.cascade(y, tests[r], partner[r], t)
                    res[(m, x, s, r)] = out
                    for c in CONFIGS:
                        for b in B:
                            D.add(('c', m, x, s, c, r), b, preds[c][b])
                assert abs(t-ref[s]['targets'][m][str(x)]['t']) < 1e-12, (m, x, s)
                for c in CONFIGS:
                    got = statistics.mean(100*res[(m, x, s, r)]['configs'][c]['mean_f1'] for r in RUNS)
                    assert abs(got-ref[s]['targets'][m][str(x)]['over_runs']['configs'][c]['mean_f1']['mean']) < 1e-9, (m, x, s, c)
                cells.setdefault((m, x), {})[s] = t
    # RULESFS10_2000 cascade of PPLX v1.2 at 0.5 %
    lab_tests = {r: cs.test_rows(lambda b: cs.ROOT/cs.SERIES['RULESFS10_fixed'][1][r]/'pplx_v12/RULESFS10_2000'/b/'test.jsonl', ids) for r in RUNS}
    t_lab = {'valid': valsel_eval.tune_risk([(float(rw['probability_match']), int(l), b) for b in B for rw, l in lv.valid_rows('RULESFS10_2000', b)], cs.X)}
    t_lab['sol200'] = t_lab['sol2000'] = valsel_eval.tune_risk(cs.solvalid_pairs('pplx_v12', cs.SOLVALID_NS, 'RULESFS10_2000'), cs.X)
    for s in SOURCES:
        assert abs(t_lab[s]-ref[s]['table11']['rulesfs10_2000']['t']) < 1e-12, s
        for r in RUNS:
            out, preds = cs.cascade(y, lab_tests[r], partner[r], t_lab[s])
            res[('lab', cs.X, s, r)] = out
            for c in CONFIGS:
                for b in B:
                    D.add(('c', 'lab', cs.X, s, c, r), b, preds[c][b])
    D.compute()
    draws = RunDraws()
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': 10_000,
           'units': 'diff, ci95 and mean_f1 in F1 points; usd at list prices; deferred shares as fractions',
           'valid_vs_solset': []}
    for m in cs.DM4+['lab']:
        for x in (cs.TARGETS if m != 'lab' else (cs.X,)):
            for sol in ('sol2000', 'sol200'):
                for c in CONFIGS:
                    spec = {b: ((('dm', m), [('c', m, x, 'valid', c, r) for r in RUNS]), (('dm', m), [('c', m, x, sol, c, r) for r in RUNS])) for b in B}
                    sg = compare(D, draws, spec, None)
                    mean = {s: statistics.mean(100*res[(m, x, s, r)]['configs'][c]['mean_f1'] for r in RUNS) for s in ('valid', sol)}
                    usd = {s: statistics.mean(res[(m, x, s, r)]['configs'][c]['usd_list'] for r in RUNS) for s in ('valid', sol)}
                    dsh = {s: statistics.mean(res[(m, x, s, r)]['deferred_share'] for r in RUNS) for s in ('valid', sol)}
                    assert abs(sg['diff']-(mean['valid']-mean[sol])) < 1e-9, (m, x, sol, c)
                    out['valid_vs_solset'].append({
                        'model': 'pplx_v12 RULESFS10_2000' if m == 'lab' else m, 'target_risk': x, 'sol_set': sol, 'config': c,
                        't_valid': t_lab['valid'] if m == 'lab' else cells[(m, x)]['valid'], 't_sol': t_lab[sol] if m == 'lab' else cells[(m, x)][sol],
                        'mean_f1_valid': mean['valid'], 'mean_f1_sol': mean[sol], 'diff': sg['diff'],
                        'fixed_ci95': sg['fixed_ci95'], 'multi_ci95': sg['multi_ci95'],
                        'fixed_significant': sg['fixed_ci95'][0] > 0 or sg['fixed_ci95'][1] < 0, 'multi_significant': sg['multi_significant'],
                        'usd_valid': usd['valid'], 'usd_sol': usd[sol], 'usd_diff': usd['valid']-usd[sol],
                        'deferred_share_valid': dsh['valid'], 'deferred_share_sol': dsh[sol], 'deferred_share_diff': dsh['valid']-dsh[sol],
                        'pairing': sg['pairing']})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    for e in out['valid_vs_solset']:
        if e['sol_set'] == 'sol2000':
            print(e['model'], e['target_risk'], e['config'], round(e['diff'], 2), [round(v, 2) for v in e['fixed_ci95']],
                  [round(v, 2) for v in e['multi_ci95']], round(e['usd_diff'], 2), flush=True)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
