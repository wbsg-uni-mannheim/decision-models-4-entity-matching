"""Offline descriptive-value token Jaccard; freeze dev thresholds before test."""
import re
import unicodedata
import numpy as np

import analyze
from expanded_calibration import PROJECT, read, threshold
from prepare import write_frozen
from protocol import BENCHMARKS, canonical


def tokens(record):
    def values(value):
        if isinstance(value, dict):
            return ' '.join(values(v) for v in value.values())
        if isinstance(value, list):
            return ' '.join(values(v) for v in value)
        return '' if value is None else str(value)
    return set(re.findall(r'\w+', unicodedata.normalize('NFKC', values(record)).casefold()))


def scores(cases):
    output = []
    for c in cases:
        left, right = (tokens(c['records'][s]) for s in ['left', 'right'])
        output.append(len(left & right)/len(left | right) if left | right else 0.)
    return np.array(output)


def main():
    selected = {}
    for b in BENCHMARKS:
        cases = read(PROJECT/'data'/b/'route.json')
        selected[b] = threshold([c['label'] for c in cases], scores(cases))
    write_frozen(PROJECT/'configs/similarity_selection.json', {
        'method': 'Descriptive values only; Unicode NFKC, casefold, Unicode word token sets; empty pair similarity zero',
        'registration': 'Post-registered extension; no test-based threshold selection',
        'threshold_source': 'D_route positive-class F1; ties nearest .5 then larger threshold', 'thresholds': selected})
    rows, intervals = [], []
    for b in [*BENCHMARKS, 'billiger-de-50un']:
        cases = read(PROJECT/'data'/b/'test.json')
        y, p = np.array([c['label'] for c in cases]), scores(cases)
        source = 'billiger-de' if b == 'billiger-de-50un' else b
        for regime, t in [('fixed_0.5_heuristic', .5), ('development_F1', selected[source])]:
            rows.append({'benchmark': b, 'regime': regime, 'threshold': t, 'development_source': source,
                         **analyze.metric(y, p >= t), 'api_cost_usd': 0.})
        intervals.append({'benchmark': b, 'comparison': 'dev-selected minus fixed .5 Jaccard',
                          **analyze.bootstrap_delta(cases, y, p >= selected[source], p >= .5)})
    out = PROJECT/'artifacts/results/similarity_baseline'
    out.mkdir(parents=True, exist_ok=True)
    analyze.csv_save(out/'metrics.csv', rows)
    (out/'metrics.json').write_text(canonical(rows)+'\n')
    (out/'intervals.json').write_text(canonical(intervals)+'\n')
    print('Similarity baseline complete:', len(rows), 'cells', flush=True)


if __name__ == '__main__':
    main()
