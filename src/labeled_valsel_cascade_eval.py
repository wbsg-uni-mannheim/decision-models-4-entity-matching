"""PPLX v1.2: labelled-data variant selected per benchmark on validation, and the cascade threshold re-tuned on it.
Read-only, free (the validation runs are in runs/labeledch_valid_run01_20261101).

Configurations: B0 and the six labelled-chapter variants RULES_<n>, FS10_<n>, RULESFS10_<n> (n = 200, 2000; rules of
Astra run 1). Validation pool: official validation sets without rules_train (30,080 pairs); B0 from
promptval_run01_20261027, the variants from labeledch_valid_run01_20261101.
a) validation F1 per benchmark for all 7 configurations (journal predictions);
b) selection per benchmark: highest validation F1, ties to the cheaper configuration (order of the test-journal cost
   per 1M pairs: B0, RULES_200, RULES_2000, FS10_200, FS10_2000, RULESFS10_200, RULESFS10_2000). Test F1 of the
   selection from the existing test runs (B0: prompt12_run01_20261019; RULES_<n>: labeledch_rules_run0{1,2,3}_20261028,
   reported per run and as the 3-run mean; FS10_<n>: labeledch_fewshot2_run01_20261031; RULESFS10_<n>:
   labeledch_rulesfs_run01_20261031);
c) cascade: t = valsel_eval.tune_risk on the pooled validation outputs of the selected configurations (each benchmark
   its own selection), target risks 0.25 / 0.5 / 1 %, applied unchanged to the selected configurations' test outputs
   (rules: run 1; runs 2 and 3 as a check with the same t). Partners as in cascade_b0_valsel_eval.py: ensemble_full
   majority (Sol, Opus, Argon) and Sol alone, existing decisions on every test pair. Cost: decision model on all test
   pairs plus the partner on the deferred pairs at list prices, also per 1M test pairs.
   Paired bootstrap (stratified per benchmark, 10,000 replicates, seed 0, the same resample for all arms): the selection
   alone and each cascade against PPLX B0 alone, and each cascade against the B0 cascade at the same risk and partner
   (variant A of cascade_b0_valsel_20261030.json, recomputed here and checked against that file).
No test pair enters selection or tuning.

    .venv/bin/python src/labeled_valsel_cascade_eval.py --out artifacts/results/labeled_valsel_cascade_20261101.json
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import protocol
import cascade_b0_valsel_eval as cv

ROOT = cv.ROOT
M = 'pplx_v12'
VALID_NS = 'labeledch_valid_run01_20261101'
ORDER = ['B0', 'RULES_200', 'RULES_2000', 'FS10_200', 'FS10_2000', 'RULESFS10_200', 'RULESFS10_2000']
RISKS = (.0025, .005, .01)
PARTNERS = ('majority', 'sol61')
B0_CASCADE = ROOT/'artifacts/results/cascade_b0_valsel_20261030.json'


def test_file(v, b, run=1):
    if v == 'B0':
        return cv.test_path(M, 'B0', b)
    ns = {'RULES': f'labeledch_rules_run0{run}_20261028', 'FS10': 'labeledch_fewshot2_run01_20261031',
          'RULESFS10': 'labeledch_rulesfs_run01_20261031'}[v.rsplit('_', 1)[0]]
    return ROOT/'runs'/ns/M/v/b/'test.jsonl'


def valid_rows(v, b):
    """Journal rows on the validation pool, in data/<b>/valid.json order without rules_train."""
    skip = {r['pair_id'] for r in json.loads((ROOT/'data'/b/'rules_train.json').read_text())}
    cases = [c for c in json.loads((ROOT/'data'/b/'valid.json').read_text()) if c['pair_id'] not in skip]
    path = ROOT/'runs'/(cv.VALID_NS if v == 'B0' else VALID_NS)/M/v/b/'valid.jsonl'
    rows = {r['pair_id']: r for r in cv.lines(path)}
    assert set(rows) >= {c['pair_id'] for c in cases} and all(rows[c['pair_id']]['probability_match'] is not None for c in cases), (v, b)
    if v != 'B0':
        assert len(rows) == len(cases), (v, b)
    return [(rows[c['pair_id']], c['label']) for c in cases]


def test_outputs(D, configs, run=1):
    """cv.test_outputs with one configuration per benchmark."""
    out = {}
    for b in protocol.BENCHMARKS:
        rows = {r['pair_id']: r for r in cv.lines(test_file(configs[b], b, run))}
        assert rows.keys() == set(D.ids[b]) and all(r['probability_match'] is not None for r in rows.values()), (configs[b], b)
        rs = [rows[i] for i in D.ids[b]]
        out[b] = {'p': np.array([r['probability_match'] for r in rs]), 'pred': np.array([r['prediction'] for r in rs], dtype=np.int8),
                  'usd': sum(cv.attempt_cost(r) for r in rs), 'missing_cost': sum(cv.unpriced(r) for r in rs), 'prompt': configs[b]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    D = cv.Data()
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'model': M, 'replicates': cv.REPS,
           'units': 'F1, rates and risks as fractions; bootstrap diff and ci95 in F1 points; USD at list prices',
           'test_pairs': D.pairs}

    # a) validation F1
    val, vf1 = {}, {}
    for v in ORDER:
        for b in protocol.BENCHMARKS:
            rows = valid_rows(v, b)
            y = np.array([l for _, l in rows], dtype=np.int8)
            pred = np.array([r['prediction'] for r, _ in rows], dtype=np.int8)
            p = np.array([r['probability_match'] for r, _ in rows])
            val[(v, b)] = [(float(pi), int(yi), b) for pi, yi in zip(p, y)]
            vf1.setdefault(v, {})[b] = {'f1': cv.f1(y, pred), 'pairs': len(rows),
                                        'p_ge_0.5_differs_from_prediction': int(((p >= .5).astype(np.int8) != pred).sum()),
                                        'usd': sum(cv.attempt_cost(r) for r, _ in rows)}
    out['validation'] = {'pairs': sum(vf1['B0'][b]['pairs'] for b in protocol.BENCHMARKS),
                         'f1': {v: {b: x['f1'] for b, x in vf1[v].items()} for v in ORDER},
                         'mean_f1': {v: float(np.mean([x['f1'] for x in vf1[v].values()])) for v in ORDER},
                         'details': vf1,
                         'run_usd': sum(x['usd'] for v in ORDER[1:] for x in vf1[v].values())}

    # b) selection and its test F1
    sel = {b: max(ORDER, key=lambda v: (vf1[v][b]['f1'], -ORDER.index(v))) for b in protocol.BENCHMARKS}
    test_f1 = {}
    for v in ORDER:
        runs = [test_outputs(D, {b: v for b in protocol.BENCHMARKS}, r) for r in ((1, 2, 3) if v.startswith('RULES_') else (1,))]
        for b in protocol.BENCHMARKS:
            per = [cv.f1(D.y[b], t[b]['pred']) for t in runs]
            test_f1.setdefault(v, {})[b] = {'runs': per, 'mean': float(np.mean(per))}
    out['selection'] = {b: {'config': sel[b], 'valid_f1': vf1[sel[b]][b]['f1'], 'test_f1_run1': test_f1[sel[b]][b]['runs'][0],
                            'test_f1_mean_runs': test_f1[sel[b]][b]['mean'], 'test_runs': len(test_f1[sel[b]][b]['runs'])}
                        for b in protocol.BENCHMARKS}
    out['selection_mean_test_f1'] = {'run1': float(np.mean([x['test_f1_run1'] for x in out['selection'].values()])),
                                     'mean_runs': float(np.mean([x['test_f1_mean_runs'] for x in out['selection'].values()]))}
    out['test_f1_all_configs'] = test_f1
    print('selection', {b: sel[b] for b in sel}, round(100*out['selection_mean_test_f1']['mean_runs'], 2), flush=True)

    # c) cascades
    b0 = json.loads(B0_CASCADE.read_text())['variants']['A_b0'][M]
    arms, out['cascade'] = {}, {}
    pooled = {'B0': [x for b in protocol.BENCHMARKS for x in val[('B0', b)]],
              'selected': [x for b in protocol.BENCHMARKS for x in val[(sel[b], b)]]}
    for name, configs in (('B0', {b: 'B0' for b in protocol.BENCHMARKS}), ('selected', sel)):
        runs = (1, 2, 3) if name == 'selected' and any(v.startswith('RULES_') for v in sel.values()) else (1,)
        for run in runs:
            test = test_outputs(D, configs, run)
            info = {'configs': configs, **cv.cost_block(test), 'levels': {}}
            for x in RISKS:
                res, preds, _ = cv.level(D, pooled[name], test, x, PARTNERS)
                info['levels'][str(x)] = res
                if run == 1:
                    for c, per in preds.items():
                        arms[(name, x, c)] = per
                if name == 'B0':
                    old = b0['levels'][str(x)]
                    assert abs(res['t']-old['t']) < 1e-12 and res['test']['deferred'] == old['test']['deferred'], x
                    for c in ('alone',)+PARTNERS:
                        assert abs(res['configs'][c]['mean_f1']-old['configs'][c]['mean_f1']) < 1e-12, (x, c)
            key = name if run == 1 else f'{name}_rules_run{run}'
            out['cascade'][key] = info
            print(key, {x: (round(r['t'], 4), round(100*r['test']['deferral_rate'], 1),
                            {c: round(100*k['mean_f1'], 2) for c, k in r['configs'].items()}) for x, r in info['levels'].items()}, flush=True)
    if 'selected_rules_run2' in out['cascade']:
        out['cascade']['selected_mean_rules_runs'] = {
            str(x): {c: float(np.mean([out['cascade'][k]['levels'][str(x)]['configs'][c]['mean_f1']
                                       for k in ('selected', 'selected_rules_run2', 'selected_rules_run3')]))
                     for c in ('alone',)+PARTNERS} for x in RISKS}

    rng = np.random.default_rng(0)
    means = {a: np.zeros(cv.REPS) for a in arms}
    for b in protocol.BENCHMARKS:
        y = D.y[b]
        idx = rng.integers(0, len(y), size=(cv.REPS, len(y)))
        for a, per in arms.items():
            code = (2*y+per[b]).astype(np.int8)[idx]
            tp, fp, fn = (code == 3).sum(1), (code == 1).sum(1), (code == 2).sum(1)
            means[a] += 2*tp/np.maximum(2*tp+fp+fn, 1)/len(protocol.BENCHMARKS)

    def point(a):
        name, x, c = a
        return out['cascade'][name]['levels'][str(x)]['configs'][c]['mean_f1']

    def interval(a, ref):
        d = 100*(means[a]-means[ref])
        return {'diff': 100*(point(a)-point(ref)), 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    ref_alone = ('B0', RISKS[0], 'alone')
    out['bootstrap'] = {'selected_alone_vs_b0_alone': interval(('selected', RISKS[0], 'alone'), ref_alone),
                        'cascade_vs_b0_alone': {str(x): {c: interval(('selected', x, c), ref_alone) for c in PARTNERS} for x in RISKS},
                        'cascade_vs_b0_cascade': {str(x): {c: interval(('selected', x, c), ('B0', x, c)) for c in PARTNERS} for x in RISKS},
                        'b0_cascade_vs_b0_alone': {str(x): {c: interval(('B0', x, c), ref_alone) for c in PARTNERS} for x in RISKS}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
