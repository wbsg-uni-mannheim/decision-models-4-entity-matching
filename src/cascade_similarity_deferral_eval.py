"""Similarity-deferral baseline for the t = 0.05 cascades of Table VII (Chris, relayed 8 Nov). Read-only, no inference.

Score of a record pair: (cosine + jaccard) / 2 from artifacts/results/similarity_scores_20261108.json (cosine of the two record
embeddings, Jaccard of their token sets [a-z0-9]+; no labels, no model outputs). Reciprocal rank fusion needs ranks within a set,
and BM25 needs a corpus, so the fused retrieval score of the examples is replaced by this absolute per-pair mix of the same two
kinds of evidence, embedding and string similarity.
Thresholds: fixed and global, from the pooled validation pairs of the eight benchmarks (official validation sets without rules_train),
set by quantile only and applied unchanged to the test sets. Target share of selected pairs: 0.13 (as requested) and 0.16 (the t = 0.05
cascades of the four models defer 16.3 % of the test pairs on average; 0.13 is about PPLX v1.2 alone).
Variants: 'high' defers the pairs with score >= the (1 - share) quantile; 'middle' defers the band between the (0.5 - share/2) and
(0.5 + share/2) quantiles.
Cascade: first-stage model decides the pairs not deferred (p >= 0.5), the partner (Sol or the majority of Sol, Opus, Argon) decides the
deferred pairs, model run r with partner run r. Models Jev 1.13, PPLX v1.2, Clef, Luna Dec. with B0. Cost as Table VII: model on all
pairs plus partner list cost on the deferred pairs.
Comparisons: real cascade (t = 0.05, cascade_fixed_t_20261107.json, asserted) minus similarity baseline, and similarity baseline minus
model alone; fixed_ci95 paired bootstrap on the run mean (10,000 replicates, seed 0, pairs drawn per benchmark in protocol order),
multi_ci95 runs drawn with replacement (rng seed 1) jointly for both arms. Random baseline next to it from
cascade_random_deferral_t005_20261107.json (it defers the same number of pairs as the real cascade).

    .venv/bin/python src/cascade_similarity_deferral_eval.py --out artifacts/results/cascade_similarity_deferral_20261108.json
"""
import argparse
import hashlib
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
SHARES = (.13, .16)
SCORES = RES/'similarity_scores_20261108.json'
FIXED = RES/'cascade_fixed_t_20261107.json'
RANDOM = RES/'cascade_random_deferral_t005_20261107.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sig(e):
    return {'diff': e['diff'], 'fixed_ci95': e['fixed_ci95'], 'multi_ci95': e['multi_ci95'],
            'fixed_significant': e['fixed_ci95'][0] > 0 or e['fixed_ci95'][1] < 0, 'multi_significant': e['multi_significant'], 'pairing': e['pairing']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    fixed = {(x['model'], x['partner']): x for x in json.loads(FIXED.read_text())['cells'] if x['t'] == T}
    rnd = {(x['model'], x['partner']): x for x in json.loads(RANDOM.read_text())['cells']}
    raw = json.loads(SCORES.read_text())
    D = BootData()
    ids, y = D.ids, D.y
    assert all(raw[b]['test'].keys() == set(ids[b]) for b in B)
    val = {b: np.array([sum(v)/2 for v in raw[b]['valid'].values()]) for b in B}
    score = {b: np.array([sum(raw[b]['test'][i])/2 for i in ids[b]]) for b in B}
    pooled = np.concatenate([val[b] for b in B])
    thresholds, masks, val_share = {}, {}, {}
    for s in SHARES:
        hi = float(np.quantile(pooled, 1-s))
        lo, up = float(np.quantile(pooled, .5-s/2)), float(np.quantile(pooled, .5+s/2))
        thresholds[f'high_{s}'] = {'low': hi, 'high': None}
        thresholds[f'middle_{s}'] = {'low': lo, 'high': up}
        masks[f'high_{s}'] = {b: score[b] >= hi for b in B}
        masks[f'middle_{s}'] = {b: (score[b] >= lo) & (score[b] <= up) for b in B}
        val_share[f'high_{s}'] = {b: float((val[b] >= hi).mean()) for b in B}
        val_share[f'middle_{s}'] = {b: float(((val[b] >= lo) & (val[b] <= up)).mean()) for b in B}
    n_all = sum(len(v) for v in ids.values())
    dec1 = cv.ensemble()
    partner = {r: cs.partner_run(r, ids, dec1)[0] for r in RUNS}
    res, sims = {}, {}
    for m in cs.DM4:
        for r in RUNS:
            test = cs.test_rows(lambda b: b0_path(m, b, r), ids)
            res[(m, r)], preds = cs.cascade(y, test, partner[r], T)
            decision_usd = res[(m, r)]['decision_usd']
            alone = {b: (test[b]['p'] >= .5).astype(np.int8) for b in B}
            wrong_total = sum(int((alone[b] != y[b]).sum()) for b in B)
            for b in B:
                D.add(('alone', m, r), b, alone[b])
                for c in CONFIGS:
                    D.add(('real', m, c, r), b, preds[c][b])
            for v, mask in masks.items():
                kept = n_d = kept_wrong = wrong_d = 0
                for b in B:
                    d = mask[b]
                    wrong = alone[b] != y[b]
                    n_d += int(d.sum()); kept += int((~d).sum()); kept_wrong += int(wrong[~d].sum()); wrong_d += int(wrong[d].sum())
                for c in CONFIGS:
                    pb, usd, errors, correct = {}, decision_usd, 0, 0
                    for b in B:
                        pc, cost = partner[r][c][b]
                        d = mask[b]
                        pred = np.where(d, pc, alone[b]).astype(np.int8)
                        D.add(('sim', v, m, c, r), b, pred)
                        pb[b] = cv.f1(y[b], pred)
                        usd += float(cost[d].sum()); errors += int((pred != y[b]).sum()); correct += int((pc[d] == y[b][d]).sum())
                    sims[(v, m, c, r)] = {'per_benchmark': pb, 'mean_f1': statistics.mean(pb.values()), 'errors': errors, 'usd_list': usd,
                                          'deferred': n_d, 'deferred_share': n_d/n_all, 'selective_risk': kept_wrong/kept,
                                          'error_share_in_deferred': wrong_d/wrong_total, 'partner_accuracy_on_deferred': correct/n_d}
    D.compute()
    draws = RunDraws()
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': 10_000,
           'inputs_sha256': {p.name: sha(p) for p in (SCORES, FIXED, RANDOM)},
           'units': 'per_run F1 as fractions; mean F1, differences and intervals in points; shares as fractions; USD at list prices',
           'thresholds': thresholds, 'score': '(cosine + jaccard) / 2',
           'test_share_selected': {v: {'pooled': float(sum(int(m[b].sum()) for b in B)/n_all), 'per_benchmark': {b: float(m[b].mean()) for b in B}} for v, m in masks.items()},
           'validation_share_selected': {v: {'pooled': float(sum((val_share[v][b]*len(val[b]) for b in B))/len(pooled)), 'per_benchmark': val_share[v]} for v in masks},
           'cells': []}
    for m in cs.DM4:
        for c in CONFIGS:
            real_f1 = statistics.mean(100*res[(m, r)]['configs'][c]['mean_f1'] for r in RUNS)
            assert abs(real_f1-fixed[(m, c)]['mean_f1']['mean']) < 1e-9, (m, c)
            for v in masks:
                rs = [sims[(v, m, c, r)] for r in RUNS]

                def spec(a, z):
                    return {b: ((('dm', m), [a(r) for r in RUNS]), (('dm', m), [z(r) for r in RUNS])) for b in B}
                e_real = compare(D, draws, spec(lambda r: ('real', m, c, r), lambda r: ('sim', v, m, c, r)), None)
                e_alone = compare(D, draws, spec(lambda r: ('sim', v, m, c, r), lambda r: ('alone', m, r)), None)
                out['cells'].append({
                    'model': m, 'partner': c, 'variant': v, 'per_run': {str(r): sims[(v, m, c, r)] for r in RUNS},
                    'mean_f1': cs.ms(100*x['mean_f1'] for x in rs), 'per_benchmark': {b: cs.ms(100*x['per_benchmark'][b] for x in rs) for b in B},
                    'usd_list': cs.ms(x['usd_list'] for x in rs), 'deferred_share': cs.ms(x['deferred_share'] for x in rs),
                    'selective_risk': cs.ms(x['selective_risk'] for x in rs), 'error_share_in_deferred': cs.ms(x['error_share_in_deferred'] for x in rs),
                    'partner_accuracy_on_deferred': cs.ms(x['partner_accuracy_on_deferred'] for x in rs),
                    'real': {'mean_f1': fixed[(m, c)]['mean_f1'], 'per_benchmark': fixed[(m, c)]['per_benchmark'], 'usd_list': fixed[(m, c)]['usd_list'],
                             'deferred_share': fixed[(m, c)]['deferred_share']},
                    'random': {'mean_f1': rnd[(m, c)]['over_runs']['mean_f1'], 'per_benchmark': rnd[(m, c)]['over_runs']['per_benchmark'],
                               'usd_list': rnd[(m, c)]['over_runs']['usd_list'], 'deferred_share': rnd[(m, c)]['over_runs']['deferred_share']},
                    'alone_mean_f1': fixed[(m, c)]['alone_mean_f1'], 'real_minus_similarity': sig(e_real), 'similarity_minus_alone': sig(e_alone)})
                x = out['cells'][-1]
                print(m, c, v, 'sim', round(x['mean_f1']['mean'], 2), 'usd', round(x['usd_list']['mean'], 2), 'def', round(100*x['deferred_share']['mean'], 1),
                      '| real', round(x['real']['mean_f1']['mean'], 2), round(x['real']['usd_list']['mean'], 2), '| rand', round(x['random']['mean_f1']['mean'], 2),
                      '| real-sim', round(e_real['diff'], 2), [round(u, 2) for u in e_real['fixed_ci95']], [round(u, 2) for u in e_real['multi_ci95']],
                      '| sim-alone', round(e_alone['diff'], 2), [round(u, 2) for u in e_alone['fixed_ci95']], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1, default=lambda o: o.item())
    print(json.dumps({'test_share_selected': out['test_share_selected'], 'thresholds': thresholds}), flush=True)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
