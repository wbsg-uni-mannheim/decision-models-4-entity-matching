"""Cascade on the full test sets: decision model with its selected prompt accepts a pair when
min(p, 1-p) <= t (t tuned on validation), the ensemble majority decides the deferred pairs.
Read-only, no inference.

    PYTHONPATH=src:artifacts/operations python3 src/cascade_eval.py > cascade.json
"""
import json
from pathlib import Path

import cascade_cost
import escalation
import inputs_study
import protocol

ROOT = Path(__file__).resolve().parents[1]
NS = 'escalation_run01_20261020'
LEVELS = [('jev', .0025), ('jev', .005), ('jev', .01), ('pplx_v12', .0025), ('pplx_v12', .005), ('pplx_v12', .01),
          ('clef', .005), ('luna_decisions', .005)]


def f1(tp, fp, fn):
    return 2*tp/(2*tp+fp+fn) if tp else 0.


def counts(pairs):
    return (sum(p == 1 and y == 1 for p, y in pairs), sum(p == 1 and y == 0 for p, y in pairs),
            sum(p == 0 and y == 1 for p, y in pairs))


def main():
    inputs_study.load_selection(ROOT)
    sets, thresholds = escalation.deferred_sets(ROOT)
    votes, majority = {}, {}
    for b in protocol.BENCHMARKS:
        for m in escalation.RUN_MODELS:
            for line in (ROOT/'runs'/NS/m/'E0'/b/'escalation.jsonl').read_text().splitlines():
                r = json.loads(line)
                votes.setdefault((b, r['pair_id']), {})[m] = r['prediction']
    for k, v in votes.items():
        assert set(v) == set(escalation.RUN_MODELS), ('incomplete votes', k)
        majority[k] = int(sum(v.values()) >= 2)
    out = {'thresholds': thresholds, 'levels': {}}
    for model, x in LEVELS:
        key = f'{model}@{x}'
        deferred = sets[key]
        assert deferred <= majority.keys(), key
        per = {}
        for b, pid, p, y in cascade_cost.test_rows(model):
            alone = int(p >= .5)
            casc = majority[(b, pid)] if (b, pid) in deferred else alone
            per.setdefault(b, {'alone': [], 'cascade': []})
            per[b]['alone'].append((alone, y)); per[b]['cascade'].append((casc, y))
        res = {'deferred': len(deferred), 'deferred_share': len(deferred)/sum(len(v['alone']) for v in per.values()),
               'per_benchmark': {b: {s: 100*f1(*counts(v[s])) for s in ('alone', 'cascade')} for b, v in per.items()}}
        for s in ('alone', 'cascade'):
            res[f'macro_{s}'] = sum(v[s] for v in res['per_benchmark'].values())/len(per)
            allp = [q for v in per.values() for q in v[s]]
            res[f'pooled_{s}'] = 100*f1(*counts(allp))
            res[f'errors_{s}'] = sum(p != y for p, y in allp)
        out['levels'][key] = res
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
