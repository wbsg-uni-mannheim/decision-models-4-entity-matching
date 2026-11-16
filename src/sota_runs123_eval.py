"""GPT-6.1 Sol, Claude Opus 5.5 and Gemini 4 Argon over three runs on all test pairs (Table 4). Read-only, free.

Run 1: ensemble_full (escalation_run01_20261020 + reference_rest_run01_20261021, loaded as
cascade_b0_valsel_eval.ensemble, i.e. paper scripts/import_ensemble_full.py). Runs 2 and 3 (Aaron, 3 Nov): Sol through the
OpenAI Batch API and Opus through the OpenRouter Batch API (runs/sota_run0<r>_20261103/<model>/results*.jsonl, the
valid row per pair), Argon through OpenRouter Google AI Studio flex (runs/sota_run0<r>_20261103/argon4/E0/<b>/test.jsonl).
Same request content as Run 1 (checked before the launch).
Per model: F1 per benchmark and run, mean and sample SD (ddof=1) over the three runs, the mean F1 over the
benchmarks per run and its mean and SD (format of sources/baseline.json f1[model][benchmark][run]); Run 1 uses the
ensemble_full predictions (cascade_b0_valsel_eval.ensemble) and is compared with its macro F1. Agreement: share of pairs
with the same decision for every pair of runs and in all three runs. Cost per run: paid (Sol: tokens at batch
prices; Opus: batch-level cost; Argon: attempt costs on flex) and list (batch and flex doubled; Run 1 as recorded).

    .venv/bin/python src/sota_runs123_eval.py --out artifacts/results/sota_runs123_20261104.json
"""
import argparse
import glob
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import protocol
import cascade_b0_valsel_eval as cv

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['sol61', 'opus55', 'argon4']
RUNS = (1, 2, 3)


def f1(y, p):
    return cv.f1(y, p)


def later_run(m, r, ids):
    """{benchmark: (predictions, paid USD)} for run 2 or 3."""
    ns = ROOT/f'runs/sota_run0{r}_20261103'
    pred, paid = {}, 0.
    if m == 'argon4':
        for b in protocol.BENCHMARKS:
            rows = {x['pair_id']: x for x in cv.lines(ns/'argon4/E0'/b/'test.jsonl')}
            assert rows.keys() == set(ids[b]) and all(x['prediction'] in (0, 1) for x in rows.values()), (m, r, b)
            pred[b] = np.array([rows[i]['prediction'] for i in ids[b]], dtype=np.int8)
            paid += sum(cv.attempt_cost(x) for x in rows.values())
        return pred, paid, 2*paid
    got = {}
    for path in sorted(glob.glob(str(ns/m/'results*.jsonl'))):
        for x in cv.lines(path):
            if 'prediction' in x:
                got.setdefault((x['benchmark'], x['pair_id']), x)
    for path in sorted(glob.glob(str(ns/m/'summary*.json'))):
        paid += json.loads(Path(path).read_text())['cost_usd']
    for b in protocol.BENCHMARKS:
        assert all((b, i) in got for i in ids[b]), (m, r, b, sum((b, i) not in got for i in ids[b]))
        pred[b] = np.array([got[(b, i)]['prediction'] for i in ids[b]], dtype=np.int8)
    return pred, paid, 2*paid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    dec = cv.ensemble()
    ids, y = {}, {}
    for b in protocol.BENCHMARKS:
        cases = json.loads((ROOT/'data'/b/'test.json').read_text())
        ids[b] = [c['pair_id'] for c in cases]
        y[b] = np.array([c['label'] for c in cases], dtype=np.int8)
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(),
           'units': 'F1 and agreement as fractions; mean_f1 and sd in F1 points; USD', 'f1': {}, 'summary': {},
           'agreement': {}, 'cost': {}}
    for m in MODELS:
        preds, cost = {1: {b: np.array([dec[m][(b, i)][0] for i in ids[b]], dtype=np.int8) for b in protocol.BENCHMARKS}}, {}
        cost['1'] = {'list_usd': sum(v[1] for v in dec[m].values())}
        for r in (2, 3):
            preds[r], paid, listed = later_run(m, r, ids)
            cost[str(r)] = {'paid_usd': paid, 'list_usd': listed}
        out['f1'][m] = {b: {str(r): f1(y[b], preds[r][b]) for r in RUNS} for b in protocol.BENCHMARKS}
        means = {str(r): 100*statistics.mean(out['f1'][m][b][str(r)] for b in protocol.BENCHMARKS) for r in RUNS}
        out['summary'][m] = {'mean_f1_per_run': means, 'mean_f1': statistics.mean(means.values()), 'sd': statistics.stdev(means.values()),
                             'per_benchmark': {b: {'mean': statistics.mean(v.values()), 'sd': statistics.stdev(v.values())}
                                               for b, v in out['f1'][m].items()},
                             'errors_per_run': {str(r): int(sum((preds[r][b] != y[b]).sum() for b in protocol.BENCHMARKS)) for r in RUNS}}
        n = sum(len(ids[b]) for b in protocol.BENCHMARKS)
        same = lambda r, s: sum(int((preds[r][b] == preds[s][b]).sum()) for b in protocol.BENCHMARKS)/n
        out['agreement'][m] = {'1_2': same(1, 2), '1_3': same(1, 3), '2_3': same(2, 3),
                               'all_three': sum(int(((preds[1][b] == preds[2][b]) & (preds[2][b] == preds[3][b])).sum()) for b in protocol.BENCHMARKS)/n}
        out['cost'][m] = cost
        print(m, {k: round(v, 2) for k, v in means.items()}, 'sd', round(out['summary'][m]['sd'], 2), out['agreement'][m], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
