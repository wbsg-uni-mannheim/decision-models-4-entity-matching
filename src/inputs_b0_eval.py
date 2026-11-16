"""Scores the input variants with B0 (inputsb0_run01_20261027) against B0 Run 1. Read-only, no inference.

Per model, benchmark and variant: F1 of the variant and of B0 Run 1 (match class), their difference and
the flip rate (share of test pairs whose decision differs from B0 Run 1). Per model and variant: paired
bootstrap of the F1 difference averaged over the variant's benchmarks (stratified per benchmark, 10,000
replicates, seed 0, same resample for B0 and the variant; method of import_inputs_bootstrap.py with B0
Run 1 instead of F0), plus correct->wrong and wrong->correct counts. Run-to-run reference: flip rate
between B0 Run 1 and B0 Run 2 per model and benchmark.
Cells whose F0 was B0 come from inputs_run01_20261019 (identical requests, checked at freeze).

    .venv/bin/python src/inputs_b0_eval.py --out artifacts/results/inputs_b0_20261028.json
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import inputs_b0
import protocol

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'runs/inputsb0_run01_20261027'
REPS = 10_000
B0_RUN1 = {'pplx_v12': 'runs/prompt12_run01_20261019'}
B0_RUN2 = {'pplx_v12': 'runs/pplx12_b0_run02_20261021'}


def journal(path, ids):
    pred = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line); pred[r['pair_id']] = r['prediction']
    assert pred.keys() == set(ids), path
    return np.array([pred[i] for i in ids], dtype=np.int8)


def f1(y, p, idx):
    yy, pp = y[idx], p[idx]
    tp = ((yy == 1) & (pp == 1)).sum(axis=-1); fp = ((yy == 0) & (pp == 1)).sum(axis=-1); fn = ((yy == 1) & (pp == 0)).sum(axis=-1)
    return 2*tp/np.maximum(2*tp+fp+fn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    inputs_b0.register_models()
    reused = inputs_b0.reused(ROOT)
    rng = np.random.default_rng(0)
    data = {}
    for b in protocol.BENCHMARKS:
        cases = json.loads((ROOT/'data'/b/'test.json').read_text())
        ids = [c['pair_id'] for c in cases]
        data[b] = (ids, np.array([c['label'] for c in cases], dtype=np.int8), rng.integers(0, len(ids), size=(REPS, len(ids))))
    out = {'description': __doc__, 'replicates': REPS, 'variants': inputs_b0.VARIANT_IDS, 'cells': {}, 'bootstrap': {},
           'run_to_run_b0': {}, 'created_at': datetime.now(timezone.utc).isoformat()}
    for m in inputs_b0.RUN_MODELS:
        b0, b0_2 = {}, {}
        for b in protocol.BENCHMARKS:
            ids, y, _ = data[b]
            b0[b] = journal(ROOT/B0_RUN1.get(m, 'runs/quality')/m/'B0'/b/'test.jsonl', ids)
            b0_2[b] = journal(ROOT/B0_RUN2.get(m, 'replications/run02/runs/quality')/m/'B0'/b/'test.jsonl', ids)
            full = np.arange(len(ids))
            out['run_to_run_b0'].setdefault(m, {})[b] = {'flip': float((b0[b] != b0_2[b]).mean()),
                                                        'f1_run1': float(f1(y, b0[b], full)), 'f1_run2': float(f1(y, b0_2[b], full))}
        for v in inputs_b0.VARIANT_IDS:
            bs = inputs_b0.benchmarks(v)
            diff, point, worse, better = np.zeros(REPS), 0., 0, 0
            for b in bs:
                ids, y, sample = data[b]
                path = reused.get((m, v, b), ROOT/NAMESPACE/m/v/b/'test.jsonl')
                var = journal(path, ids)
                full = np.arange(len(ids))
                fv, f0 = float(f1(y, var, full)), float(f1(y, b0[b], full))
                out['cells'].setdefault(m, {}).setdefault(v, {})[b] = {'f1': fv, 'f1_b0': f0, 'diff': fv-f0,
                                                                       'flip': float((var != b0[b]).mean()), 'n': len(ids),
                                                                       'reused': (m, v, b) in reused}
                point += (fv-f0)/len(bs)
                diff += (f1(y, var, sample)-f1(y, b0[b], sample))/len(bs)
                worse += int(((b0[b] == y) & (var != y)).sum()); better += int(((b0[b] != y) & (var == y)).sum())
            out['bootstrap'].setdefault(m, {})[v] = {'diff': 100*point, 'ci95': [100*float(np.percentile(diff, 2.5)),
                                                                                  100*float(np.percentile(diff, 97.5))],
                                                    'correct_to_wrong': worse, 'wrong_to_correct': better, 'benchmarks': len(bs)}
        print(m, {v: round(out['bootstrap'][m][v]['diff'], 2) for v in inputs_b0.VARIANT_IDS}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
