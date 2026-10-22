"""Ensemble quality on escalated test pairs (sample or full run). Read-only, no inference.

Majority vote of the three members per pair. Compared on the same pairs with each
member, Luna generative (selected prompt) and the decision models (selected prompt,
decision at p >= 0.5), against the benchmark labels.

    PYTHONPATH=src:artifacts/operations python3 src/escalation_eval.py [--sample]
"""
import argparse
import json
from pathlib import Path

import escalation
import inputs_study
import protocol

ROOT = Path(__file__).resolve().parents[1]


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def selected(model, b):
    p = inputs_study.SELECTED[model][b]
    own = p != 'B0' or model == 'pplx_v12'
    return {r['pair_id']: r['prediction']
            for r in lines(ROOT/'runs'/('prompt12_run01_20261019' if own else 'quality')/model/p/b/'test.jsonl')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', action='store_true')
    args = ap.parse_args()
    split, ns = ('escalation_sample', 'escalation_sample_run01_20261020') if args.sample else ('escalation', 'escalation_run01_20261020')
    inputs_study.load_selection(ROOT)
    sets, _ = escalation.deferred_sets(ROOT)
    rows = []
    for b in protocol.BENCHMARKS:
        cases = json.loads((ROOT/'data'/b/(split+'.json')).read_text())
        votes = {m: {r['pair_id']: r['prediction'] for r in lines(ROOT/'runs'/ns/m/'E0'/b/(split+'.jsonl'))}
                 for m in escalation.RUN_MODELS}
        others = {m: selected(m, b) for m in ['luna', 'jev', 'pplx_v12', 'clef', 'luna_decisions']}
        for c in cases:
            v = {m: votes[m].get(c['pair_id']) for m in escalation.RUN_MODELS}
            rows.append({'b': b, 'id': c['pair_id'], 'y': c['label'], 'votes': v,
                         'majority': int(sum(v.values()) >= 2) if None not in v.values() else None,
                         **{m: others[m][c['pair_id']] for m in others}})
    complete = [r for r in rows if r['majority'] is not None]
    out = {'pairs': len(rows), 'complete': len(complete), 'matches': sum(r['y'] for r in complete)}

    def acc(key, subset):
        return 100*sum(r[key] == r['y'] if key in r else r['votes'][key] == r['y'] for r in subset)/len(subset) if subset else None
    for name, keys in [('all', None)] + [(k, v) for k, v in sets.items()]:
        subset = [r for r in complete if keys is None or (r['b'], r['id']) in keys]
        out[name] = {'n': len(subset), 'majority': acc('majority', subset),
                     **{m: acc(m, subset) for m in escalation.RUN_MODELS},
                     **{m: acc(m, subset) for m in ['luna', 'jev', 'pplx_v12', 'clef', 'luna_decisions']},
                     'unanimous_share': 100*sum(len(set(r['votes'].values())) == 1 for r in subset)/len(subset) if subset else None}
    by_b = {}
    for r in complete:
        by_b.setdefault(r['b'], []).append(r)
    out['by_benchmark'] = {b: {'n': len(v), 'majority': acc('majority', v), 'luna': acc('luna', v)} for b, v in by_b.items()}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
