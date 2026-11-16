"""Random-deferral baseline for the cascade table, three runs. Read-only, no inference (Aaron, relayed 6 Nov: "Okay bitte mach
einmal die random kaskade 3 mal wie alles andere").

Jev 1.13, PPLX v1.2, Clef, Luna Dec. with B0; partners Sol and the majority of Sol, Opus, Argon. In run r the random cascade
defers as many pairs per benchmark as the real cascade of that model and run (t at 0.5 % from the Sol 2,000 set,
cascade_sol2000_runs123_20261105_v2.json), drawn uniformly; the other pairs keep the model's p >= 0.5. Model run r goes with
partner run r. Both partners of a model use the same deferred pairs, as in the real cascade.
Draw: per run and benchmark one permutation, rng = default_rng([RUN_SEED[r], benchmark index in protocol order]); a model
defers its first d pairs, so the models share the draw and their deferred sets are nested.
Per cell: F1 per benchmark, mean F1 per run, mean and sample SD over the runs; errors; list cost (the number of deferred
pairs equals the real cascade, the cost differs slightly because the partner cost differs per pair); error share and
partner accuracy on the deferred pairs. real_minus_random and random_minus_alone: paired bootstrap on the run mean
(fixed_ci95; 10,000 replicates, seed 0, pairs drawn per benchmark in protocol order) and multi-bootstrap (multi_ci95; runs
drawn jointly for both arms), as in cascade_sol2000_tests_20261105.json. These intervals hold the main draw fixed.
seed_sensitivity: the three-run mean F1 over N_DRAWS further draws (rng = default_rng([RUN_SEED[r], benchmark index, i])):
mean, SD, min, max, 2.5 and 97.5 percentile, the same for real minus random, and the share of draws with real <= random.

--fixed-t (7 Nov, Aaron: chapter 5 reports the cascades without labeled pairs at a fixed t = 0.05): the real cascade uses this
fixed t instead of the tuned one and is checked against cascade_fixed_t_20261107.json; everything else unchanged.

    .venv/bin/python src/cascade_random_deferral_eval.py --out artifacts/results/cascade_random_deferral_20261106.json
    .venv/bin/python src/cascade_random_deferral_eval.py --fixed-t 0.05 --out artifacts/results/cascade_random_deferral_t005_20261107.json
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
RUN_SEED = {1: 20261101, 2: 20261102, 3: 20261103}
N_DRAWS = 1000


def f1_counts(tp, fp, fn):
    return 2*tp/np.maximum(2*tp+fp+fn, 1)


def sig(e):
    return {'diff': e['diff'], 'fixed_ci95': e['fixed_ci95'], 'multi_ci95': e['multi_ci95'],
            'fixed_significant': e['fixed_ci95'][0] > 0 or e['fixed_ci95'][1] < 0, 'multi_significant': e['multi_significant'],
            'pairing': e['pairing']}


def spread(x):
    return {'mean': float(np.mean(x)), 'sd': float(np.std(x, ddof=1)), 'min': float(np.min(x)), 'max': float(np.max(x)),
            'p2_5': float(np.percentile(x, 2.5)), 'p97_5': float(np.percentile(x, 97.5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fixed-t', type=float)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    ref = json.loads((RES/'cascade_sol2000_runs123_20261105_v2.json').read_text())
    fixed = {(x['model'], x['partner']): x for x in json.loads((RES/'cascade_fixed_t_20261107.json').read_text())['cells'] if x['t'] == args.fixed_t}
    cs.TUNING[0] = 'sol2000'
    D = BootData()
    ids, y = D.ids, D.y
    dec1 = cv.ensemble()
    partner = {r: cs.partner_run(r, ids, dec1)[0] for r in RUNS}
    perm = {(r, b): np.random.default_rng([RUN_SEED[r], i]).permutation(len(ids[b])) for r in RUNS for i, b in enumerate(B)}
    cells, sens = {}, {}
    for m in cs.DM4:
        t = args.fixed_t if args.fixed_t is not None else valsel_eval.tune_risk(cs.tuning_pairs(m), cs.X)
        assert args.fixed_t is not None or abs(t-ref['table9'][m]['t']) < 1e-12, m
        per = {c: {} for c in CONFIGS}
        real, draws_f1 = {}, {c: np.zeros((N_DRAWS, len(RUNS))) for c in CONFIGS}
        for ri, r in enumerate(RUNS):
            test = cs.test_rows(lambda b: b0_path(m, b, r), ids)
            real[r], preds = cs.cascade(y, test, partner[r], t)
            if args.fixed_t is None:
                assert real[r]['deferred'] == ref['table9'][m]['per_run'][str(r)]['deferred'], (m, r)
            else:
                assert real[r]['deferred'] == fixed[(m, 'sol61')]['per_run'][str(r)]['deferred'], (m, r)
            counts, wrong_total = {}, 0
            res = {c: {'pb': {}, 'errors': 0, 'usd': real[r]['decision_usd'], 'correct': 0} for c in CONFIGS}
            wrong_d = 0
            for bi, b in enumerate(B):
                p, yb = test[b]['p'], y[b]
                alone = (p >= .5).astype(np.int8)
                counts[b] = int((np.minimum(p, 1-p) > t).sum())
                dmask = np.zeros(len(p), dtype=bool); dmask[perm[(r, b)][:counts[b]]] = True
                wrong = alone != yb
                wrong_total += int(wrong.sum()); wrong_d += int(wrong[dmask].sum())
                D.add(('alone', m, r), b, alone)
                for c in CONFIGS:
                    pc, usd = partner[r][c][b]
                    pred = np.where(dmask, pc, alone).astype(np.int8)
                    D.add(('rand', m, c, r), b, pred)
                    D.add(('real', m, c, r), b, preds[c][b])
                    res[c]['pb'][b] = cv.f1(yb, pred)
                    res[c]['errors'] += int((pred != yb).sum())
                    res[c]['usd'] += float(usd[dmask].sum())
                    res[c]['correct'] += int((pc[dmask] == yb[dmask]).sum())
                    # further draws: F1 from count differences, only the deferred positions change
                    pos, neg = yb == 1, yb == 0
                    a_tp, a_fp, a_fn = int((alone[pos] == 1).sum()), int((alone[neg] == 1).sum()), int((alone[pos] == 0).sum())
                    dtp = (pos & (pc == 1)).astype(np.int32)-(pos & (alone == 1)).astype(np.int32)
                    dfp = (neg & (pc == 1)).astype(np.int32)-(neg & (alone == 1)).astype(np.int32)
                    dfn = -dtp
                    for i in range(N_DRAWS):
                        pick = perm_draw(r, bi, i, len(p))[:counts[b]]
                        draws_f1[c][i, ri] += float(f1_counts(a_tp+dtp[pick].sum(), a_fp+dfp[pick].sum(), a_fn+dfn[pick].sum()))/len(B)
            for c in CONFIGS:
                per[c][str(r)] = {'seed': RUN_SEED[r], 'deferred_per_benchmark': counts, 'deferred': sum(counts.values()),
                                  'deferred_share': sum(counts.values())/sum(len(v) for v in ids.values()),
                                  'per_benchmark': res[c]['pb'], 'mean_f1': statistics.mean(res[c]['pb'].values()), 'errors': res[c]['errors'],
                                  'usd_list': res[c]['usd'], 'error_share_in_deferred': wrong_d/wrong_total,
                                  'partner_accuracy_on_deferred': res[c]['correct']/sum(counts.values())}
        cells[m] = {'t': t, 'per': per, 'real': real}
        sens[m] = draws_f1
        print(m, 'draws done', flush=True)
    D.compute()
    rd = RunDraws()
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': 10_000, 'tuning_source': 'fixed' if args.fixed_t is not None else 'sol2000',
           'target_risk': None if args.fixed_t is not None else cs.X, 'fixed_t': args.fixed_t, 'run_seeds': RUN_SEED, 'n_seed_draws': N_DRAWS,
           'units': 'per_run F1 as fractions; over_runs F1, differences, intervals and seed_sensitivity in F1 points; shares as fractions; USD at list prices',
           'cells': []}
    for m in cs.DM4:
        for c in CONFIGS:
            pr = cells[m]['per'][c]
            rs = list(pr.values())
            real = cells[m]['real']
            spec_real = {b: ((('dm', m), [('real', m, c, r) for r in RUNS]), (('dm', m), [('rand', m, c, r) for r in RUNS])) for b in B}
            spec_alone = {b: ((('dm', m), [('rand', m, c, r) for r in RUNS]), (('dm', m), [('alone', m, r) for r in RUNS])) for b in B}
            e_real, e_alone = compare(D, rd, spec_real, None), compare(D, rd, spec_alone, None)
            f_real = statistics.mean(100*real[r]['configs'][c]['mean_f1'] for r in RUNS)
            f_rand = statistics.mean(100*x['mean_f1'] for x in rs)
            f_alone = statistics.mean(100*real[r]['configs']['alone']['mean_f1'] for r in RUNS)
            if args.fixed_t is None:
                got = ref['table9'][m]['over_runs']['configs']
                assert abs(f_real-got[c]['mean_f1']['mean']) < 1e-9 and abs(f_alone-got['alone']['mean_f1']['mean']) < 1e-9, (m, c)
            else:
                assert abs(f_real-fixed[(m, c)]['mean_f1']['mean']) < 1e-9 and abs(f_alone-fixed[(m, c)]['alone_mean_f1']) < 1e-9, (m, c)
            assert abs(e_real['diff']-(f_real-f_rand)) < 1e-9 and abs(e_alone['diff']-(f_rand-f_alone)) < 1e-9, (m, c)
            u_rand = statistics.mean(x['usd_list'] for x in rs)
            u_real = statistics.mean(real[r]['configs'][c]['usd_list'] for r in RUNS)
            d3 = 100*sens[m][c].mean(axis=1)
            out['cells'].append({
                'model': m, 'partner': c, 't': cells[m]['t'], 'per_run': pr,
                'over_runs': {'mean_f1': cs.ms(100*x['mean_f1'] for x in rs),
                              'per_benchmark': {b: cs.ms(100*x['per_benchmark'][b] for x in rs) for b in B},
                              'errors': cs.ms(x['errors'] for x in rs), 'usd_list': cs.ms(x['usd_list'] for x in rs),
                              'deferred_share': cs.ms(x['deferred_share'] for x in rs),
                              'error_share_in_deferred': cs.ms(x['error_share_in_deferred'] for x in rs),
                              'partner_accuracy_on_deferred': cs.ms(x['partner_accuracy_on_deferred'] for x in rs)},
                'mean_f1_real': f_real, 'mean_f1_alone': f_alone, 'usd_real': u_real, 'usd_random_minus_real': u_rand-u_real,
                'real_minus_random': sig(e_real), 'random_minus_alone': sig(e_alone),
                'seed_sensitivity': {'three_run_mean_f1': spread(d3), 'real_minus_random': spread(f_real-d3),
                                     'random_minus_alone': spread(d3-f_alone), 'share_draws_real_not_better': float((f_real <= d3).mean())}})
            x = out['cells'][-1]
            print(m, c, 'random', round(f_rand, 2), 'real', round(f_real, 2), 'alone', round(f_alone, 2),
                  'real-rand', round(e_real['diff'], 2), [round(v, 2) for v in e_real['fixed_ci95']], [round(v, 2) for v in e_real['multi_ci95']],
                  'rand-alone', round(e_alone['diff'], 2), [round(v, 2) for v in e_alone['fixed_ci95']],
                  'seed sd', round(x['seed_sensitivity']['three_run_mean_f1']['sd'], 3), 'usd', round(u_rand, 2), round(u_real, 2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1, default=lambda o: o.item())
    print('wrote', args.out)


PERM_CACHE = {}


def perm_draw(r, bi, i, n):
    """Permutation of draw i, cached so all models and partners share the draws."""
    key = (r, bi, i)
    if key not in PERM_CACHE:
        PERM_CACHE[key] = np.random.default_rng([RUN_SEED[r], bi, i]).permutation(n).astype(np.int32)
    return PERM_CACHE[key]


if __name__ == '__main__':
    main()
