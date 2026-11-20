"""Similarity deferral near the decision boundary: the similarity baseline for the t = 0.05 cascades of Table VII.
Read-only, no inference.

Scores: cosine and Jaccard from artifacts/results/similarity_scores_20261108.json, kept as two separate lists (no averaging).
Thresholds: per benchmark and metric (16 values) on the official validation split without rules_train, with gold labels. With k = the
number of validation matches, t is the score of the k-th most similar validation pair (validation pairs sorted by score descending,
ties by pair_id). FP and FN by rank: non-matches among the k top-ranked pairs = matches outside them, equal by construction. Also
logged: the count-based FP (non-matches with score >= t) and FN (matches with score < t) and the number of pairs tied at t.
N: per model, run and benchmark, the number of pairs the real cascade defers, min(p, 1-p) > 0.05 (counts[b] as in
cascade_random_deferral_eval.py).
Selection on test, per benchmark: per metric the test pairs are sorted ascending by score (ties by pair_id); t is located at
pos = the first rank with score >= t (searchsorted, side='left'). The metric's candidates alternate between the nearest unused rank
above (pos, pos+1, ...) and the nearest unused rank below (pos-1, pos-2, ...), above first; when one side runs out, the other side
continues. The two metrics take turns at every pick, cosine first; a metric whose next candidate is already taken advances on its
own list until it finds a new pair. This gives one fixed order of all test pairs per benchmark; a cell defers the first N pairs.
No random draws; only N differs across models and runs. Both partners of a model use the same deferred pairs.
Cascade and comparisons as cascade_similarity_deferral_eval.py: model p >= 0.5 on the kept pairs, partner (Sol or the majority of Sol,
Opus, Argon) on the deferred pairs, model run r with partner run r; cost = model on all pairs plus partner list cost on the deferred
pairs. real_minus_boundary and boundary_minus_alone: fixed_ci95 paired bootstrap on the run mean (10,000 replicates, seed 0, pairs
drawn per benchmark in protocol order) and multi_ci95 (runs drawn with replacement, rng seed 1, jointly for both arms).
Validation check (no tuning): the same selection on the validation pairs with a band of round(share x n_valid) pairs, share = the mean
deferred share of the four models and three runs, per benchmark ('per_benchmark') and pooled over the benchmarks ('pooled');
reported: match share in the band.

    .venv/bin/python src/cascade_similarity_boundary_eval.py --gate
    .venv/bin/python src/cascade_similarity_boundary_eval.py --out artifacts/results/cascade_similarity_boundary_20261117.json
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

B, RUNS, RES, ROOT = cs.B, cs.RUNS, cs.RES, cs.ROOT
T = .05
CONFIGS = ('sol61', 'majority')
METRICS = ('cosine', 'jaccard')
SCORES = RES/'similarity_scores_20261108.json'
FIXED = RES/'cascade_fixed_t_20261107.json'
RANDOM = RES/'cascade_random_deferral_t005_20261107.json'
OLD_SIM = RES/'cascade_similarity_deferral_20261108.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sig(e):
    return {'diff': e['diff'], 'fixed_ci95': e['fixed_ci95'], 'multi_ci95': e['multi_ci95'],
            'fixed_significant': e['fixed_ci95'][0] > 0 or e['fixed_ci95'][1] < 0, 'multi_significant': e['multi_significant'], 'pairing': e['pairing']}


def valid_cases(b):
    """Official validation split without rules_train, as labeled_valid_prep_20261101.valid_cases."""
    skip = {r['pair_id'] for r in json.loads((ROOT/'data'/b/'rules_train.json').read_text())}
    return [c for c in json.loads((ROOT/'data'/b/'valid.json').read_text()) if c['pair_id'] not in skip]


def threshold(score, label, pids):
    """t = score of the k-th most similar pair (k = matches); FP/FN by rank and by count; ties at t."""
    k = int(label.sum())
    order = sorted(range(len(score)), key=lambda i: (-score[i], pids[i]))
    t = float(score[order[k-1]])
    top = np.zeros(len(score), dtype=bool)
    top[order[:k]] = True
    return {'t': t, 'k': k, 'n': len(score), 'fp_rank': int((top & (label == 0)).sum()), 'fn_rank': int((~top & (label == 1)).sum()),
            'fp_count': int(((score >= t) & (label == 0)).sum()), 'fn_count': int(((score < t) & (label == 1)).sum()),
            'tied_at_t': int((score == t).sum())}


def metric_order(score, pids, t):
    """Ranks alternating above/below t, nearest first, above first; one side continues when the other runs out."""
    asc = sorted(range(len(score)), key=lambda i: (score[i], pids[i]))
    srt = np.array([score[i] for i in asc])
    pos = int(np.searchsorted(srt, t, side='left'))
    above, below, out = pos, pos-1, []
    turn = 0
    while above < len(asc) or below >= 0:
        if (turn == 0 and above < len(asc)) or below < 0:
            out.append(asc[above]); above += 1
        else:
            out.append(asc[below]); below -= 1
        turn ^= 1
    return out, pos


def merged_order(scores, pids, ts):
    """Cosine and Jaccard take turns at every pick (cosine first); taken pairs are skipped within the metric's own list."""
    lists = [metric_order(scores[s], pids, ts[s])[0] for s in METRICS]
    ptr, taken, out = [0, 0], set(), []
    n = len(pids)
    turn = 0
    while len(out) < n:
        li = lists[turn]
        while ptr[turn] < n and li[ptr[turn]] in taken:
            ptr[turn] += 1
        if ptr[turn] < n:
            i = li[ptr[turn]]; ptr[turn] += 1
            taken.add(i); out.append(i)
        turn ^= 1
    assert len(set(out)) == n
    return np.array(out, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path)
    ap.add_argument('--gate', action='store_true', help='thresholds and validation band check only, no output file')
    args = ap.parse_args()
    assert args.gate or (args.out and not args.out.exists()), args.out
    raw = json.loads(SCORES.read_text())
    fixed = {(x['model'], x['partner']): x for x in json.loads(FIXED.read_text())['cells'] if x['t'] == T}
    D = BootData()
    ids, y = D.ids, D.y
    assert all(raw[b]['test'].keys() == set(ids[b]) for b in B)
    thresholds, val_order, val_y, test_order, test_pos = {}, {}, {}, {}, {}
    for b in B:
        vc = valid_cases(b)
        vp = [c['pair_id'] for c in vc]
        assert set(vp) == raw[b]['valid'].keys() and len(vp) == len(set(vp)), b
        val_y[b] = np.array([int(c['label']) for c in vc], dtype=np.int8)
        vs = {s: np.array([raw[b]['valid'][p][j] for p in vp]) for j, s in enumerate(METRICS)}
        ts = {s: np.array([raw[b]['test'][i][j] for i in ids[b]]) for j, s in enumerate(METRICS)}
        thresholds[b] = {s: threshold(vs[s], val_y[b], vp) for s in METRICS}
        for s in METRICS:
            t = thresholds[b][s]['t']
            thresholds[b][s].update({'test_at_or_above_t': int((ts[s] >= t).sum()), 'test_tied_at_t': int((ts[s] == t).sum()), 'test_n': len(ids[b])})
        tt = {s: thresholds[b][s]['t'] for s in METRICS}
        val_order[b] = merged_order(vs, vp, tt)
        test_order[b] = merged_order(ts, list(ids[b]), tt)
        test_pos[b] = {s: metric_order(ts[s], list(ids[b]), tt[s])[1] for s in METRICS}
    # deferral counts of the real cascade
    counts = {}
    for m in cs.DM4:
        for r in RUNS:
            test = cs.test_rows(lambda b: b0_path(m, b, r), ids)
            counts[(m, r)] = {b: int((np.minimum(test[b]['p'], 1-test[b]['p']) > T).sum()) for b in B}
            assert sum(counts[(m, r)].values()) == fixed[(m, 'sol61')]['per_run'][str(r)]['deferred'], (m, r)
    n_all = sum(len(ids[b]) for b in B)
    share_b = {b: statistics.mean(counts[k][b]/len(ids[b]) for k in counts) for b in B}
    share_pooled = statistics.mean(sum(counts[k].values())/n_all for k in counts)
    band = {}
    for name, sh in (('per_benchmark', share_b), ('pooled', {b: share_pooled for b in B})):
        band[name] = {}
        for b in B:
            n = int(round(sh[b]*len(val_y[b])))
            band[name][b] = {'share': sh[b], 'n_band': n, 'match_share': float(val_y[b][val_order[b][:n]].mean()) if n else None,
                             'match_share_all_valid': float(val_y[b].mean())}
    print('thresholds', json.dumps({b: {s: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in thresholds[b][s].items()} for s in METRICS} for b in B}, indent=1), flush=True)
    for name in band:
        print('validation band', name, {b: (round(100*band[name][b]['match_share'], 1), band[name][b]['n_band']) for b in B}, flush=True)
    if args.gate:
        return
    rnd = {(x['model'], x['partner']): x for x in json.loads(RANDOM.read_text())['cells']}
    old = {}
    for x in json.loads(OLD_SIM.read_text())['cells']:
        old.setdefault((x['model'], x['partner']), {})[x['variant']] = {'mean_f1': x['mean_f1'], 'usd_list': x['usd_list'], 'deferred_share': x['deferred_share']}
    dec1 = cv.ensemble()
    partner = {r: cs.partner_run(r, ids, dec1)[0] for r in RUNS}
    res, bnd = {}, {}
    for m in cs.DM4:
        for r in RUNS:
            test = cs.test_rows(lambda b: b0_path(m, b, r), ids)
            res[(m, r)], preds = cs.cascade(y, test, partner[r], T)
            alone = {b: (test[b]['p'] >= .5).astype(np.int8) for b in B}
            wrong_total = sum(int((alone[b] != y[b]).sum()) for b in B)
            masks = {}
            for b in B:
                d = np.zeros(len(ids[b]), dtype=bool)
                d[test_order[b][:counts[(m, r)][b]]] = True
                assert int(d.sum()) == counts[(m, r)][b]
                masks[b] = d
                D.add(('alone', m, r), b, alone[b])
                for c in CONFIGS:
                    D.add(('real', m, c, r), b, preds[c][b])
            n_d = sum(int(masks[b].sum()) for b in B)
            kept = n_all-n_d
            kept_wrong = sum(int((alone[b] != y[b])[~masks[b]].sum()) for b in B)
            wrong_d = sum(int((alone[b] != y[b])[masks[b]].sum()) for b in B)
            for c in CONFIGS:
                pb, usd, errors, correct = {}, res[(m, r)]['decision_usd'], 0, 0
                for b in B:
                    pc, cost = partner[r][c][b]
                    d = masks[b]
                    pred = np.where(d, pc, alone[b]).astype(np.int8)
                    D.add(('boundary', m, c, r), b, pred)
                    pb[b] = cv.f1(y[b], pred)
                    usd += float(cost[d].sum()); errors += int((pred != y[b]).sum()); correct += int((pc[d] == y[b][d]).sum())
                bnd[(m, c, r)] = {'deferred_per_benchmark': counts[(m, r)], 'per_benchmark': pb, 'mean_f1': statistics.mean(pb.values()),
                                  'errors': errors, 'usd_list': usd, 'deferred': n_d, 'deferred_share': n_d/n_all, 'selective_risk': kept_wrong/kept,
                                  'error_share_in_deferred': wrong_d/wrong_total, 'partner_accuracy_on_deferred': correct/n_d}
    D.compute()
    draws = RunDraws()
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': 10_000, 't': T,
           'inputs_sha256': {**{p.name: sha(p) for p in (SCORES, FIXED, RANDOM, OLD_SIM)},
                             **{f'{b}/{f}': sha(ROOT/'data'/b/f) for b in B for f in ('valid.json', 'rules_train.json')}},
           'units': 'per_run F1 as fractions; mean F1, differences and intervals in points; shares as fractions; USD at list prices',
           'thresholds': thresholds, 'test_position_of_t': test_pos, 'validation_band_check': band, 'cells': []}
    for m in cs.DM4:
        for c in CONFIGS:
            rs = [bnd[(m, c, r)] for r in RUNS]
            real_f1 = statistics.mean(100*res[(m, r)]['configs'][c]['mean_f1'] for r in RUNS)
            assert abs(real_f1-fixed[(m, c)]['mean_f1']['mean']) < 1e-9, (m, c)
            f_b = statistics.mean(100*x['mean_f1'] for x in rs)
            alone_f1 = statistics.mean(100*res[(m, r)]['configs']['alone']['mean_f1'] for r in RUNS)

            def spec(a, z):
                return {b: ((('dm', m), [a(r) for r in RUNS]), (('dm', m), [z(r) for r in RUNS])) for b in B}
            e_real = compare(D, draws, spec(lambda r: ('real', m, c, r), lambda r: ('boundary', m, c, r)), None)
            e_alone = compare(D, draws, spec(lambda r: ('boundary', m, c, r), lambda r: ('alone', m, r)), None)
            assert abs(e_real['diff']-(real_f1-f_b)) < 1e-9 and abs(e_alone['diff']-(f_b-alone_f1)) < 1e-9, (m, c)
            out['cells'].append({
                'model': m, 'partner': c, 'per_run': {str(r): bnd[(m, c, r)] for r in RUNS},
                'mean_f1': cs.ms(100*x['mean_f1'] for x in rs), 'per_benchmark': {b: cs.ms(100*x['per_benchmark'][b] for x in rs) for b in B},
                'usd_list': cs.ms(x['usd_list'] for x in rs), 'deferred_share': cs.ms(x['deferred_share'] for x in rs),
                'selective_risk': cs.ms(x['selective_risk'] for x in rs), 'error_share_in_deferred': cs.ms(x['error_share_in_deferred'] for x in rs),
                'partner_accuracy_on_deferred': cs.ms(x['partner_accuracy_on_deferred'] for x in rs),
                'real': {'mean_f1': fixed[(m, c)]['mean_f1'], 'per_benchmark': fixed[(m, c)]['per_benchmark'], 'usd_list': fixed[(m, c)]['usd_list'],
                         'deferred_share': fixed[(m, c)]['deferred_share']},
                'random': {'mean_f1': rnd[(m, c)]['over_runs']['mean_f1'], 'usd_list': rnd[(m, c)]['over_runs']['usd_list'],
                           'deferred_share': rnd[(m, c)]['over_runs']['deferred_share']},
                'old_similarity': old[(m, c)], 'alone_mean_f1': alone_f1,
                'real_minus_boundary': sig(e_real), 'boundary_minus_alone': sig(e_alone)})
            x = out['cells'][-1]
            print(m, c, 'boundary', round(f_b, 2), 'usd', round(x['usd_list']['mean'], 2), '| real', round(real_f1, 2),
                  '| rand', round(x['random']['mean_f1']['mean'], 2), '| old sim', {v: round(o['mean_f1']['mean'], 2) for v, o in old[(m, c)].items()},
                  '| real-bnd', round(e_real['diff'], 2), [round(u, 2) for u in e_real['fixed_ci95']], [round(u, 2) for u in e_real['multi_ci95']],
                  '| bnd-alone', round(e_alone['diff'], 2), [round(u, 2) for u in e_alone['fixed_ci95']], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1, default=lambda o: o.item())
    print('wrote', args.out)


if __name__ == '__main__':
    main()
