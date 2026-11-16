"""Cascades without test selection: B0 (variant A) and the prompt selected on validation (variant B). Read-only, free.

Deferral rule of the cascade chapter (src/escalation.py, src/valsel_eval.py): the decision model keeps a pair when
min(p, 1-p) <= t and predicts a match when p >= 0.5; t = valsel_eval.tune_risk on the pooled official validation sets
without the 69 rules_train pairs, applied unchanged to test. Target risks: PPLX v1.2 and Jev 1.13 at 0.25 / 0.5 / 1 %,
Clef and Luna Dec. at 0.5 %; Kev and Clef-flash only threshold and deferral at 0.5 %.
  A: B0 Run 1 on validation (promptval_run01_20261027) and test (runs/quality; pplx_v12 its own prompt12 B0).
  B: per model and benchmark the prompt of prompt_validation_20261027.json selection[m][b].prompt; its validation
     outputs (promptval_run01_20261027) set t, its test outputs (prompt12_run01_20261019, B0 as in A) are routed.
Partners decide the deferred pairs with B0 and existing decisions on every test pair: ensemble majority of GPT-6.1 Sol,
Claude Opus 5.5 and Gemini 4 Argon (runs/escalation_run01_20261020 + runs/reference_rest_run01_20261021, loaded as in
paper scripts/import_ensemble_full.py), Sol alone, and Luna Gen. B0 Run 1 (runs/quality/luna/B0, PPLX and Jev only).
Cost as in the cascade table: decision model on all test pairs (journal attempt costs, pairs without reported cost
count as zero; pairs with such an attempt are counted) plus the partner on the deferred pairs at list prices (flex and batch attempts doubled;
Luna Gen. attempt costs as recorded), also per 1M test pairs.
Bootstrap as in inputs_b0_eval.py: stratified per benchmark in protocol.BENCHMARKS order, 10,000 replicates, seed 0,
the same resample for every arm; mean F1 over the 8 benchmarks in points with 95 % CI for each cascade against the
model alone and for variant B against variant A (same model, configuration and target risk).
Regression check before anything new: the same code with the test-selected prompts (F0) and valsel_run01_20261019
validation must reproduce the thresholds, deferral counts, F1, errors and costs of the current cascade table
(paper sources cascade.json and ensemble_full.json, passed as files). No API calls, nothing is selected on test.

    .venv/bin/python src/cascade_b0_valsel_eval.py --cascade-json <cascade.json> --ensemble-json <ensemble_full.json> \
        --out artifacts/results/cascade_b0_valsel_20261030.json
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import inputs_study
import prompt12
import protocol
import valsel_eval

ROOT = Path(__file__).resolve().parents[1]
PV = ROOT/'artifacts/results/prompt_validation_20261027.json'
VALID_NS, TEST_NS, F0_VALID_NS = 'promptval_run01_20261027', 'prompt12_run01_20261019', 'valsel_run01_20261019'
ESCALATION, REST = ROOT/'runs/escalation_run01_20261020', ROOT/'runs/reference_rest_run01_20261021'
MEMBERS = ['sol61', 'opus55', 'argon4']
RISKS = {'pplx_v12': (.0025, .005, .01), 'jev': (.0025, .005, .01), 'clef': (.005,), 'luna_decisions': (.005,)}
DEFERRAL_ONLY = {'kev': (.005,), 'clef_flash': (.005,)}
PARTNERS = {'pplx_v12': ('majority', 'sol61', 'luna'), 'jev': ('majority', 'sol61', 'luna'),
            'clef': ('majority', 'sol61'), 'luna_decisions': ('majority', 'sol61')}
CONFIGS = ('alone', 'majority', 'sol61', 'luna')
REPS = 10_000


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def attempt_cost(r):
    return sum((a.get('cost') or 0.) for a in r.get('attempts') or [])


def unpriced(r):
    """At least one attempt without reported cost (counted as zero, as in the cascade table's stage1)."""
    return any(a.get('cost') is None for a in r.get('attempts') or [])


def list_cost(m, r):
    """Recorded attempt costs, doubled on the flex and batch tiers (import_ensemble_full.list_cost)."""
    total = 0.
    for a in r.get('attempts') or []:
        tier = (a.get('response') or {}).get('service_tier') or a.get('service_tier')
        if tier is None and m == 'opus55':
            tier = 'standard'
        total += 2*(a.get('cost') or 0.) if tier in ('flex', 'batch') else a.get('cost') or 0.
    return total


def ensemble():
    """decisions[m][(b, pid)] = (prediction, list-price cost) for the three members on every test pair."""
    dec = {m: {} for m in MEMBERS}

    def add(m, r, listed):
        k = (r['benchmark'], r['pair_id'])
        assert k not in dec[m] and r['prediction'] in (0, 1), (m, k)
        dec[m][k] = (r['prediction'], listed)
    for m in MEMBERS:
        for b in protocol.BENCHMARKS:
            for r in lines(ESCALATION/m/'E0'/b/'escalation.jsonl'):
                add(m, r, list_cost(m, r))
    for b in protocol.BENCHMARKS:
        for r in lines(REST/'argon4'/'E0'/b/'reference_rest.jsonl'):
            add('argon4', r, list_cost('argon4', r))
    summary = json.loads((REST/'opus55_batch'/'summary.json').read_text())
    batches = defaultdict(list)
    for r in lines(REST/'opus55_batch'/'results.jsonl'):
        batches[r['batch_id']].append(r)
    for batch_id, rs in batches.items():
        for r in rs:
            add('opus55', r, 2*summary['status'][batch_id]['cost']/len(rs))
    for r in lines(REST/'sol61_direct'/'results.jsonl'):
        if r.get('error') is None:
            add('sol61', r, list_cost('sol61', r))
    for path in sorted(REST.glob('sol61_direct/results.fill-*.jsonl')):
        for r in lines(path):
            add('sol61', r, list_cost('sol61', r))
    return dec


def test_path(m, v, b):
    own = v != 'B0' or m in prompt12.EXTRA_MODELS
    return ROOT/'runs'/(TEST_NS if own else 'quality')/m/v/b/'test.jsonl'


def f1(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    return 2*tp/(2*tp+fp+fn) if tp else 0.


class Data:
    """Labels, partner decisions and partner list costs per benchmark, aligned to data/<b>/test.json order."""

    def __init__(self):
        dec = ensemble()
        self.ids, self.y, self.partner, self.partner_usd, self.missing = {}, {}, {}, {}, {}
        for b in protocol.BENCHMARKS:
            cases = json.loads((ROOT/'data'/b/'test.json').read_text())
            ids = [c['pair_id'] for c in cases]
            self.ids[b], self.y[b] = ids, np.array([c['label'] for c in cases], dtype=np.int8)
            for m in MEMBERS:
                assert all((b, i) in dec[m] for i in ids), (m, b)
            votes = {m: np.array([dec[m][(b, i)][0] for i in ids], dtype=np.int8) for m in MEMBERS}
            usd = {m: np.array([dec[m][(b, i)][1] for i in ids]) for m in MEMBERS}
            luna = {r['pair_id']: r for r in lines(test_path('luna', 'B0', b))}
            assert luna.keys() == set(ids) and all(r['prediction'] in (0, 1) for r in luna.values()), b
            self.partner[b] = {'majority': (sum(votes.values()) >= 2).astype(np.int8), 'sol61': votes['sol61'],
                               'luna': np.array([luna[i]['prediction'] for i in ids], dtype=np.int8)}
            self.partner_usd[b] = {'majority': sum(usd.values()), 'sol61': usd['sol61'],
                                   'luna': np.array([attempt_cost(luna[i]) for i in ids])}
            self.missing[b] = {'luna': sum(unpriced(luna[i]) for i in ids)}
        self.pairs = sum(len(v) for v in self.ids.values())
        assert all(len(dec[m]) == self.pairs for m in MEMBERS)
        self.members_usd_list = {m: sum(v[1] for v in dec[m].values()) for m in MEMBERS}


def validation(m, prompt_of, namespace, fixed_dir=None):
    """(p, y, b) on the official validation sets without rules_train; prompt_of(b) names the prompt folder."""
    out = []
    for b in protocol.BENCHMARKS:
        cases = {c['pair_id']: c['label'] for c in json.loads((ROOT/'data'/b/'valid.json').read_text())}
        skip = {r['pair_id'] for r in json.loads((ROOT/'data'/b/'rules_train.json').read_text())}
        got = {r['pair_id']: r['probability_match'] for r in lines(ROOT/'runs'/namespace/m/(fixed_dir or prompt_of(b))/b/'valid.jsonl')}
        assert got.keys() == cases.keys() and all(p is not None for p in got.values()), (m, b, len(got), len(cases))
        out += [(got[i], y, b) for i, y in cases.items() if i not in skip]
    return out


def test_outputs(D, m, prompt_of):
    """Per benchmark: probabilities, journal predictions, attempt cost and pairs without reported cost."""
    out = {}
    for b in protocol.BENCHMARKS:
        rows = {r['pair_id']: r for r in lines(test_path(m, prompt_of(b), b))}
        assert rows.keys() == set(D.ids[b]) and all(r['probability_match'] is not None for r in rows.values()), (m, b)
        rs = [rows[i] for i in D.ids[b]]
        out[b] = {'p': np.array([r['probability_match'] for r in rs]), 'pred': np.array([r['prediction'] for r in rs], dtype=np.int8),
                  'usd': sum(attempt_cost(r) for r in rs), 'missing_cost': sum(unpriced(r) for r in rs), 'prompt': prompt_of(b)}
    return out


def level(D, val, test, x, partners):
    """Threshold, deferral, risks and every configuration's predictions for one model and target risk."""
    t = valsel_eval.tune_risk(val, x)
    va = valsel_eval.accept(val, t)
    res = {'t': t, 'valid': {'pairs': len(val), 'deferral_rate': 1-va['coverage'], 'selective_risk': va['risk']}}
    preds, dmask = {c: {} for c in ('alone',)+partners}, {}
    n = acc_wrong = acc_n = err = err_def = 0
    part_correct, own_correct = defaultdict(int), 0
    stage1 = sum(test[b]['usd'] for b in protocol.BENCHMARKS)
    partner_usd = defaultdict(float)
    for b in protocol.BENCHMARKS:
        p, y = test[b]['p'], D.y[b]
        alone = (p >= .5).astype(np.int8)
        d = np.minimum(p, 1-p) > t
        dmask[b] = d
        preds['alone'][b] = alone
        for c in partners:
            preds[c][b] = np.where(d, D.partner[b][c], alone).astype(np.int8)
            part_correct[c] += int((D.partner[b][c][d] == y[d]).sum())
            partner_usd[c] += float(D.partner_usd[b][c][d].sum())
        wrong = alone != y
        n += len(y); acc_n += int((~d).sum()); acc_wrong += int(wrong[~d].sum())
        err += int(wrong.sum()); err_def += int(wrong[d].sum()); own_correct += int((~wrong[d]).sum())
    deferred = n-acc_n
    res['test'] = {'pairs': n, 'deferred': deferred, 'deferral_rate': deferred/n, 'selective_risk': acc_wrong/acc_n,
                   'errors_alone': err, 'errors_in_deferred': err_def, 'error_share_in_deferred': err_def/err if err else None,
                   'model_accuracy_on_deferred': own_correct/deferred if deferred else None,
                   'partner_accuracy_on_deferred': {c: part_correct[c]/deferred if deferred else None for c in partners}}
    res['configs'] = {}
    for c, per in preds.items():
        pb = {b: f1(D.y[b], v) for b, v in per.items()}
        usd = stage1 + (partner_usd[c] if c != 'alone' else 0.)
        res['configs'][c] = {'per_benchmark': pb, 'mean_f1': sum(pb.values())/len(pb),
                             'errors': sum(int((v != D.y[b]).sum()) for b, v in per.items()),
                             'usd_list': usd, 'usd_per_1m_pairs': usd/n*1e6, 'partner_usd_list': partner_usd[c] if c != 'alone' else 0.}
    return res, preds, dmask


def cost_block(test):
    return {'decision_usd': sum(v['usd'] for v in test.values()), 'decision_pairs_with_unpriced_attempt': sum(v['missing_cost'] for v in test.values())}


def regression(D, cascade, ens):
    """The current cascade table (test-selected F0 prompts) must come out of the same code."""
    inputs_study.load_selection(ROOT)
    F0 = inputs_study.SELECTED
    checked = {}
    for m, risks in RISKS.items():
        val = validation(m, None, F0_VALID_NS, fixed_dir='F0')
        test = test_outputs(D, m, lambda b: F0[m][b])
        stage1 = cascade['levels_detail']['stage1'][m]
        assert abs(cost_block(test)['decision_usd']-stage1['usd']) < 1e-9, m
        assert cost_block(test)['decision_pairs_with_unpriced_attempt'] == stage1['pairs_missing_cost'], m
        for x in risks:
            key = f'{m}@{x}'
            res, _, _ = level(D, val, test, x, ('majority', 'sol61'))
            old, det, new = cascade['cascade_full']['thresholds'][key], cascade['levels_detail']['levels'][key], ens['levels'][key]
            assert abs(res['t']-old) < 1e-12 and res['test']['deferred'] == det['deferred'] == new['pairs'], key
            assert abs(res['test']['selective_risk']-det['accepted_error']) < 1e-12, key
            assert res['test']['errors_in_deferred'] == det['errors_in_deferred_alone'] and res['test']['errors_alone'] == det['errors_alone'], key
            for c in ('alone', 'majority', 'sol61'):
                assert abs(res['configs'][c]['mean_f1']-new['configs'][c]['macro_f1']) < 1e-12, (key, c)
                assert res['configs'][c]['errors'] == new['configs'][c]['errors'], (key, c)
            assert abs(res['configs']['majority']['partner_usd_list']-new['ensemble_usd_list']) < 1e-9, key
            assert abs(res['configs']['sol61']['partner_usd_list']-new['members_usd_list']['sol61']) < 1e-9, key
            checked[key] = {'t': res['t'], 'deferred': res['test']['deferred'],
                            'mean_f1': {c: res['configs'][c]['mean_f1'] for c in ('alone', 'majority', 'sol61')}}
    for m in MEMBERS:
        assert abs(D.members_usd_list[m]-ens['members'][m]['usd_list']) < 1e-9, m
    return checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cascade-json', type=Path, required=True)
    ap.add_argument('--ensemble-json', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reps', type=int, default=REPS)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    D = Data()
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': args.reps,
           'units': 'F1, rates and risks as fractions; bootstrap diff and ci95 in F1 points; USD at list prices',
           'test_pairs': D.pairs,
           'regression_check_current_table': regression(D, json.loads(args.cascade_json.read_text()),
                                                        json.loads(args.ensemble_json.read_text())),
           'partner_cost': {'luna_pairs_with_unpriced_attempt': sum(v['luna'] for v in D.missing.values())},
           'variants': {}, 'deferral_only': {}}
    print('regression check passed', flush=True)
    pv = json.loads(PV.read_text())
    variants = {'A_b0': lambda m: (lambda b: 'B0'), 'B_valsel': lambda m: (lambda b: pv['selection'][m][b]['prompt'])}
    arms = {}  # (variant, model, x, config) -> {b: predictions}
    for v, prompt_for in variants.items():
        for m in list(RISKS)+list(DEFERRAL_ONLY):
            prompt_of = prompt_for(m)
            val = validation(m, prompt_of, VALID_NS)
            test = test_outputs(D, m, prompt_of)
            mismatch = sum(int(((t['p'] >= .5).astype(np.int8) != t['pred']).sum()) for t in test.values())
            if v == 'B_valsel':
                for b in protocol.BENCHMARKS:
                    assert abs(f1(D.y[b], test[b]['pred'])-pv['selection'][m][b]['test_f1']) < 1e-12, (m, b)
            info = {'prompts': {b: prompt_of(b) for b in protocol.BENCHMARKS}, **cost_block(test),
                    'pairs_where_p_ge_0.5_differs_from_journal_prediction': mismatch, 'levels': {}}
            if m in DEFERRAL_ONLY:
                for x in DEFERRAL_ONLY[m]:
                    res, _, _ = level(D, val, test, x, ())
                    info['levels'][str(x)] = {'t': res['t'], 'valid': res['valid'], 'test': res['test']}
                out['deferral_only'].setdefault(v, {})[m] = info
                continue
            for x in RISKS[m]:
                res, preds, _ = level(D, val, test, x, PARTNERS[m])
                info['levels'][str(x)] = res
                for c, per in preds.items():
                    arms[(v, m, x, c)] = per
            out['variants'].setdefault(v, {})[m] = info
            print(v, m, {x: (round(r['t'], 4), round(100*r['test']['deferral_rate'], 1),
                             {c: round(100*k['mean_f1'], 2) for c, k in r['configs'].items()}) for x, r in info['levels'].items()}, flush=True)
    rng = np.random.default_rng(0)
    means = {a: np.zeros(args.reps) for a in arms}
    for b in protocol.BENCHMARKS:
        y = D.y[b]
        idx = rng.integers(0, len(y), size=(REPS, len(y)))[:args.reps]
        for a, per in arms.items():
            code = (2*y+per[b]).astype(np.int8)[idx]  # 3 tp, 1 fp, 2 fn
            tp, fp, fn = (code == 3).sum(1), (code == 1).sum(1), (code == 2).sum(1)
            means[a] += 2*tp/np.maximum(2*tp+fp+fn, 1)/len(protocol.BENCHMARKS)
        print('bootstrap', b, flush=True)

    def interval(a, ref):
        point = 100*(out_f1(a)-out_f1(ref))
        d = 100*(means[a]-means[ref])
        return {'diff': point, 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

    def out_f1(a):
        v, m, x, c = a
        return out['variants'][v][m]['levels'][str(x)]['configs'][c]['mean_f1']
    out['bootstrap'] = {'cascade_vs_alone': {}, 'valsel_vs_b0': {}}
    for (v, m, x, c) in arms:
        if c != 'alone':
            out['bootstrap']['cascade_vs_alone'].setdefault(v, {}).setdefault(m, {}).setdefault(str(x), {})[c] = interval((v, m, x, c), (v, m, x, 'alone'))
        if v == 'B_valsel':
            out['bootstrap']['valsel_vs_b0'].setdefault(m, {}).setdefault(str(x), {})[c] = interval((v, m, x, c), ('A_b0', m, x, c))
    out['literature_pplx_v12_at_0.005'] = {v: {c: out['variants'][v]['pplx_v12']['levels']['0.005']['configs'][c]['per_benchmark']
                                               for c in ('alone', 'majority')} for v in variants}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
