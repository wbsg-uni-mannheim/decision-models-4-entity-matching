"""Does calibration on validation transfer to test? Read-only, no inference.

For each model: reliability (Brier, calibration error) on validation and test, and
selective prediction with thresholds tuned on validation only, then applied to test.
A pair is accepted when min(p, 1-p) <= t, i.e. the model is confident; the decision
stays p >= 0.5. Validation excludes the Astra training pairs (rules_train).

    PYTHONPATH=src python3 src/valsel_eval.py --models jev [--valid-ns valsel_run01_20261019]
"""
import argparse
import json
from pathlib import Path

import inputs_study
import protocol

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = (.85,)
RISK = (.0025, .005, .01)


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def pairs(root, model, split, namespace):
    """(p, y, benchmark) for the selected prompt; test F0 comes from the prompt study."""
    out = []
    for b in protocol.BENCHMARKS:
        cases = {c['pair_id']: c['label'] for c in json.loads((root/'data'/b/(split+'.json')).read_text())}
        if split == 'valid':
            skip = {r['pair_id'] for r in json.loads((root/'data'/b/'rules_train.json').read_text())}
            path = root/'runs'/namespace/model/'F0'/b/'valid.jsonl'
        else:
            skip = set()
            p = inputs_study.SELECTED[model][b]
            own = p != 'B0' or model == 'pplx_v12'
            path = root/'runs'/('prompt12_run01_20261019' if own else 'quality')/model/p/b/'test.jsonl'
        got = {r['pair_id']: r['probability_match'] for r in rows(path)}
        assert got.keys() == cases.keys(), (model, split, b, len(got), len(cases))
        out += [(got[i], y, b) for i, y in cases.items() if i not in skip]
    return out


def reliability(data):
    n = len(data)
    brier = sum((p-y)**2 for p, y, _ in data)/n
    bins = [[] for _ in range(10)]
    for p, y, _ in data:
        bins[min(int(p*10), 9)].append((p, y))
    cal = sum(len(x)*(sum(p for p, _ in x)/len(x)-sum(y for _, y in x)/len(x))**2 for x in bins if x)/n
    return {'brier': brier, 'cal': cal}


def accept(data, t):
    a = [(p, y) for p, y, _ in data if min(p, 1-p) <= t]
    wrong = sum((p >= .5) != y for p, y in a)
    total_wrong = sum((p >= .5) != y for p, y, _ in data)
    return {'coverage': len(a)/len(data), 'risk': wrong/len(a) if a else 0.,
            'errors_deferred': 1-wrong/total_wrong if total_wrong else 1.}


def tune_coverage(data, c):
    u = sorted(min(p, 1-p) for p, _, _ in data)
    return u[max(0, int(c*len(u))-1)]


def tune_risk(data, eps):
    """Largest threshold whose accepted set on validation has error <= eps."""
    best, wrong, n = 0., 0, 0
    order = sorted(data, key=lambda x: min(x[0], 1-x[0]))
    for i, (p, y, _) in enumerate(order):
        n += 1; wrong += (p >= .5) != y
        nxt = min(order[i+1][0], 1-order[i+1][0]) if i+1 < len(order) else None
        if nxt != min(p, 1-p) and wrong/n <= eps:  # only cut between distinct values
            best = min(p, 1-p)
    return best


def evaluate(root, model, namespace, per_benchmark=True):
    val, test = pairs(root, model, 'valid', namespace), pairs(root, model, 'test', namespace)
    out = {'n': {'valid': len(val), 'test': len(test)},
           'reliability': {'valid': reliability(val), 'test': reliability(test)}, 'selective': {}}
    groups = {'pooled': (val, test)}
    if per_benchmark:
        groups.update({b: ([x for x in val if x[2] == b], [x for x in test if x[2] == b]) for b in protocol.BENCHMARKS})
    for name, (v, t) in groups.items():
        res = {}
        for c in COVERAGE:
            th = tune_coverage(v, c)
            res[f'coverage_{c:.2f}'] = {'t': th, 'valid': accept(v, th), 'test': accept(t, th)}
        for eps in RISK:
            th = tune_risk(v, eps)
            res[f'risk_{eps:.4f}'] ={'t': th, 'valid': accept(v, th), 'test': accept(t, th)}
        out['selective'][name] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--valid-ns', default='valsel_run01_20261019')
    ap.add_argument('--root', type=Path, default=ROOT)
    args = ap.parse_args()
    inputs_study.load_selection(args.root)
    print(json.dumps({m: evaluate(args.root, m, args.valid_ns) for m in args.models}, indent=1))


if __name__ == '__main__':
    main()
