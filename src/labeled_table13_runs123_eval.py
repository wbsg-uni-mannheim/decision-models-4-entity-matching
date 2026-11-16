"""Table 13 (labelled data, Jev 1.13 and PPLX v1.2) over three runs against B0 Run 1. Read-only, no inference.

Series of three test runs per variant (n = 200, 2000):
- RULES_<n>: Astra rules of repetitions 1-3 (labeledch_rules_run0{1,2,3}_20261028; the rules change with the run);
- FS10_<n>: labeledch_fewshot2_run0{1,2,3}_20261031 (same retrieved examples in every run);
- RULESFS10_<n> fixed: run-1 Astra rules in every run (labeledch_rulesfs_run0{1,2,3}_20261031);
- RULESFS10_<n> varying: run k with the rules of Astra repetition k (run 1 labeledch_rulesfs_run01_20261031, runs 2 and 3
  labeledch_rulesfsrep_run0{2,3}_20261104; Aaron, 4 Nov: "Run 1 auf Astra-Regeln 1, Run 2 auf Astra-Regeln 2, ...").
Reference: B0 Run 1 (as labeled_rules_eval.py). Per model, series and size: F1 per benchmark and run, mean and sample SD
over the runs; mean F1 over the 8 benchmarks per run, its difference to B0, mean and SD over runs. Paired bootstrap as
labeled_rules_eval.py (10,000 replicates, seed 0, stratified per benchmark, the same resample for B0 and every run):
95 % CI per run and of the 3-run mean (F1 averaged over the runs per replicate). Run 1 must reproduce
labeled_rules_20261028.json, labeled_fewshot2_20261031.json and labeled_rulesfs_20261031.json (asserted).

    .venv/bin/python src/labeled_table13_runs123_eval.py --out artifacts/results/labeled_table13_runs123_20261104.json
"""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import protocol
from inputs_b0_eval import B0_RUN1, f1, journal

ROOT = Path(__file__).resolve().parents[1]
MODELS, SIZES, REPS = ['jev', 'pplx_v12'], ['200', '2000'], 10_000
SERIES = {
    'RULES': ('RULES', {r: f'runs/labeledch_rules_run0{r}_20261028' for r in (1, 2, 3)}, 'labeled_rules_20261028.json'),
    'FS10': ('FS10', {r: f'runs/labeledch_fewshot2_run0{r}_20261031' for r in (1, 2, 3)}, 'labeled_fewshot2_20261031.json'),
    'RULESFS10_fixed': ('RULESFS10', {r: f'runs/labeledch_rulesfs_run0{r}_20261031' for r in (1, 2, 3)}, 'labeled_rulesfs_20261031.json'),
    'RULESFS10_varying': ('RULESFS10', {1: 'runs/labeledch_rulesfs_run01_20261031', 2: 'runs/labeledch_rulesfsrep_run02_20261104',
                                        3: 'runs/labeledch_rulesfsrep_run03_20261104'}, 'labeled_rulesfs_20261031.json')}


def ci(x):
    return [100*float(np.percentile(x, 2.5)), 100*float(np.percentile(x, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    rng = np.random.default_rng(0)
    data = {}
    for b in protocol.BENCHMARKS:
        cases = json.loads((ROOT/'data'/b/'test.json').read_text())
        ids = [c['pair_id'] for c in cases]
        data[b] = (ids, np.array([c['label'] for c in cases], dtype=np.int8), rng.integers(0, len(ids), size=(REPS, len(ids))))
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': REPS,
           'units': 'F1 as fraction in cells; means, diff, sd and ci95 in F1 points', 'cells': {}, 'summary': {}}
    b0 = {}
    for m in MODELS:
        for b in protocol.BENCHMARKS:
            ids, y, sample = data[b]
            p = journal(ROOT/B0_RUN1.get(m, 'runs/quality')/m/'B0'/b/'test.jsonl', ids)
            b0[(m, b)] = (float(f1(y, p, np.arange(len(ids)))), f1(y, p, sample))
    for name, (prefix, runs, ref_file) in SERIES.items():
        ref = json.loads((ROOT/'artifacts/results'/ref_file).read_text())
        for m in MODELS:
            for s in SIZES:
                v = f'{prefix}_{s}'
                per_run = {r: np.zeros(REPS) for r in runs}
                f0_mean = statistics.mean(b0[(m, b)][0] for b in protocol.BENCHMARKS)
                for b in protocol.BENCHMARKS:
                    ids, y, sample = data[b]
                    cell = {}
                    for r, ns in runs.items():
                        p = journal(ROOT/ns/m/v/b/'test.jsonl', ids)
                        cell[str(r)] = float(f1(y, p, np.arange(len(ids))))
                        per_run[r] += (f1(y, p, sample)-b0[(m, b)][1])/len(protocol.BENCHMARKS)
                    old = ref['cells'][m][b][s if prefix == 'RULES' else v]['f1']
                    assert abs(cell['1']-old) < 1e-12, (name, m, b, v)
                    vals = list(cell.values())
                    out['cells'].setdefault(name, {}).setdefault(m, {}).setdefault(s, {})[b] = {
                        'f1': cell, 'mean': statistics.mean(vals), 'sd': statistics.stdev(vals), 'f1_b0': b0[(m, b)][0]}
                cells = out['cells'][name][m][s].values()
                means = {str(r): 100*statistics.mean(c['f1'][str(r)] for c in cells) for r in runs}
                out['summary'].setdefault(name, {}).setdefault(m, {})[s] = {
                    'variant': v, 'f1_b0': 100*f0_mean, 'mean_f1_per_run': means,
                    'diff_per_run': {r: x-100*f0_mean for r, x in means.items()},
                    'ci95_per_run': {str(r): ci(per_run[r]) for r in runs},
                    'mean_f1': statistics.mean(means.values()), 'sd': statistics.stdev(means.values()),
                    'diff': statistics.mean(means.values())-100*f0_mean,
                    'ci95': ci(sum(per_run.values())/len(runs))}
                print(name, m, s, {r: round(x-100*f0_mean, 2) for r, x in means.items()}, 'mean', round(statistics.mean(means.values())-100*f0_mean, 2),
                      'sd', round(statistics.stdev(means.values()), 2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
