"""Astra rules over three repetitions (sets 200 and 2000), Jev and PPLX v1.2 against B0 Run 1. Read-only, no inference.

Runs: labeledch_rules_run01_20261028 (rules of configs/labeled_chapter_rules_20261028.json) and
labeledch_rules_run0{2,3}_20261028 (Astra repetitions 2 and 3, configs/labeled_chapter_rules_20261028_rep{2,3}.json),
each RULES_200 / RULES_2000 on the full test sets. Per model, size and run: F1 per benchmark, mean F1 over the 8
benchmarks and its difference to B0; over the three runs: mean, sample SD (ddof=1) of the run means and per benchmark.
Paired bootstrap as labeled_rules_eval.py (10,000 replicates, seed 0, stratified per benchmark, same resample for B0
and all runs): 95% CI of each run's mean difference and of the 3-run mean (F1 averaged over the runs per replicate).
Run 1 must reproduce artifacts/results/labeled_rules_20261028.json.

    .venv/bin/python src/labeled_rules_reps_eval.py --out artifacts/results/labeled_rules_reps_20261031.json
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import protocol
from inputs_b0_eval import B0_RUN1, f1, journal

ROOT = Path(__file__).resolve().parents[1]
RUNS = {1: 'runs/labeledch_rules_run01_20261028', 2: 'runs/labeledch_rules_run02_20261028', 3: 'runs/labeledch_rules_run03_20261028'}
RULES = {1: 'configs/labeled_chapter_rules_20261028.json', 2: 'configs/labeled_chapter_rules_20261028_rep2.json',
         3: 'configs/labeled_chapter_rules_20261028_rep3.json'}
RUN1 = ROOT/'artifacts/results/labeled_rules_20261028.json'
SIZES, MODELS, REPS = ['200', '2000'], ['jev', 'pplx_v12'], 10_000


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    run1 = json.loads(RUN1.read_text())
    rng = np.random.default_rng(0)
    data = {}
    for b in protocol.BENCHMARKS:
        cases = json.loads((ROOT/'data'/b/'test.json').read_text())
        ids = [c['pair_id'] for c in cases]
        data[b] = (ids, np.array([c['label'] for c in cases], dtype=np.int8), rng.integers(0, len(ids), size=(REPS, len(ids))))
    out = {'description': __doc__, 'replicates': REPS, 'units': 'F1 as fraction; diff and ci95 in F1 points', 'rules': {},
           'cells': {}, 'per_run': {}, 'three_runs': {}, 'created_at': datetime.now(timezone.utc).isoformat()}
    for r, path in RULES.items():
        src = json.loads((ROOT/path).read_text())['sources']
        out['rules'][str(r)] = {'file': path, **{s: {'mean_rules': float(np.mean([v['n_rules'] for k, v in src.items() if k.endswith('/'+s)])),
                                                     'mean_chars': float(np.mean([v['chars'] for k, v in src.items() if k.endswith('/'+s)]))}
                                                 for s in SIZES}}
    for m in MODELS:
        for s in SIZES:
            v = f'RULES_{s}'
            diff = {r: np.zeros(REPS) for r in RUNS}
            mean_diff = np.zeros(REPS)
            for b in protocol.BENCHMARKS:
                ids, y, sample = data[b]
                full = np.arange(len(ids))
                b0 = journal(ROOT/B0_RUN1.get(m, 'runs/quality')/m/'B0'/b/'test.jsonl', ids)
                f0, f0_s = float(f1(y, b0, full)), f1(y, b0, sample)
                runs_s, cell = [], {'f1_b0': f0, 'f1': {}}
                for r, ns in RUNS.items():
                    pred = journal(ROOT/ns/m/v/b/'test.jsonl', ids)
                    fv, fs = float(f1(y, pred, full)), f1(y, pred, sample)
                    if r == 1:
                        assert abs(fv-run1['cells'][m][b][s]['f1']) < 1e-12, (m, b, v)
                    cell['f1'][str(r)] = fv
                    diff[r] += (fs-f0_s)/len(protocol.BENCHMARKS)
                    runs_s.append(fs)
                mean_diff += (np.mean(runs_s, axis=0)-f0_s)/len(protocol.BENCHMARKS)
                vals = list(cell['f1'].values())
                cell.update(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)), diff_mean=100*(float(np.mean(vals))-f0))
                out['cells'].setdefault(m, {}).setdefault(s, {})[b] = cell
            cells = out['cells'][m][s].values()
            f0_mean = float(np.mean([c['f1_b0'] for c in cells]))
            run_means = {r: float(np.mean([c['f1'][str(r)] for c in cells])) for r in RUNS}
            for r in RUNS:
                out['per_run'].setdefault(m, {}).setdefault(s, {})[str(r)] = {'f1': run_means[r], 'diff': 100*(run_means[r]-f0_mean),
                                                                               'ci95': [100*x for x in ci(diff[r])]}
            assert abs(run_means[1]-run1['mean'][m][s]['f1']) < 1e-12, (m, v)
            mean = float(np.mean(list(run_means.values())))
            out['three_runs'].setdefault(m, {})[s] = {'f1_b0': f0_mean, 'f1_mean': mean, 'f1_sd': float(np.std(list(run_means.values()), ddof=1)),
                                                     'diff': 100*(mean-f0_mean), 'ci95': [100*x for x in ci(mean_diff)],
                                                     'diff_sd': 100*float(np.std(list(run_means.values()), ddof=1))}
            print(m, s, {r: round(x['diff'], 2) for r, x in out['per_run'][m][s].items()},
                  'mean', round(out['three_runs'][m][s]['diff'], 2), [round(x, 2) for x in out['three_runs'][m][s]['ci95']],
                  'sd', round(out['three_runs'][m][s]['diff_sd'], 2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
