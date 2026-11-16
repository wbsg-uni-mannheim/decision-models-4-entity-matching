"""Input variants with B0 (Tables X/XI) over three runs: variant run k against B0 run k. Read-only, no inference.

Variant runs: Run 1 inputsb0_run01_20261027 (cells whose F0 was B0 from inputs_run01_20261019, as inputs_b0_eval.py);
Runs 2 and 3 (Aaron, 3 Nov): the eight variants F1-F4, R2-R4, R6 from inputsb0_run0{2,3}_20261103, the entry swap R1
from inputsb0_r1_run0{2,3}_20261104 (its repeats had not been run before). B0 run k: runs/quality and replications/run0{2,3}/runs/quality; PPLX v1.2 prompt12_run01_20261019 and
pplx12_b0_run0{2,3}_20261021.
Per model, variant, benchmark and run: F1 of the variant and of B0 run k, difference, flip rate (share of pairs whose
decision differs from B0 run k). Per model and variant: mean F1 difference over the variant's benchmarks and mean flip
rate, per run and as mean and SD (ddof=1) over the three runs. Paired bootstrap as inputs_b0_eval.py (stratified per
benchmark in protocol order, 10,000 replicates, seed 0; the same resample for every run): per run (Run 1 reproduces
artifacts/results/inputs_b0_20261028.json, asserted), significant_in_all_runs, and run_mean (per replicate the
difference averaged over the three runs). Run-to-run reference: flip rate between the B0 runs per model and benchmark.

    .venv/bin/python src/inputs_b0_runs123_eval.py --out artifacts/results/inputs_b0_runs123_20261104.json
"""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import inputs_b0
import protocol
from inputs_b0_eval import f1, journal

ROOT = Path(__file__).resolve().parents[1]
RUNS = (1, 2, 3)
REPS = 10_000
RUN1 = ROOT/'artifacts/results/inputs_b0_20261028.json'


def b0_path(m, b, r):
    if m == 'pplx_v12':
        ns = {1: 'runs/prompt12_run01_20261019', 2: 'runs/pplx12_b0_run02_20261021', 3: 'runs/pplx12_b0_run03_20261021'}[r]
    else:
        ns = {1: 'runs/quality', 2: 'replications/run02/runs/quality', 3: 'replications/run03/runs/quality'}[r]
    return ROOT/ns/m/'B0'/b/'test.jsonl'


def variant_path(reused, m, v, b, r):
    if r == 1:
        return reused.get((m, v, b), ROOT/'runs/inputsb0_run01_20261027'/m/v/b/'test.jsonl')
    ns = f'inputsb0_r1_run0{r}_20261104' if v in inputs_b0.REPEAT_VARIANTS else f'inputsb0_run0{r}_20261103'
    return ROOT/'runs'/ns/m/v/b/'test.jsonl'


def ci(x):
    return [100*float(np.percentile(x, 2.5)), 100*float(np.percentile(x, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    inputs_b0.register_models()
    reused = inputs_b0.reused(ROOT)
    run1 = json.loads(RUN1.read_text())
    rng = np.random.default_rng(0)
    data = {}
    for b in protocol.BENCHMARKS:
        cases = json.loads((ROOT/'data'/b/'test.json').read_text())
        ids = [c['pair_id'] for c in cases]
        data[b] = (ids, np.array([c['label'] for c in cases], dtype=np.int8), rng.integers(0, len(ids), size=(REPS, len(ids))))
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': REPS,
           'units': 'F1 and flip as fractions in cells; diff, flip means, sd and ci95 in points', 'variants': inputs_b0.VARIANT_IDS,
           'cells': {}, 'summary': {}, 'bootstrap': {}, 'run_to_run_b0': {}}
    for m in inputs_b0.RUN_MODELS:
        b0 = {(b, r): journal(b0_path(m, b, r), data[b][0]) for b in protocol.BENCHMARKS for r in RUNS}
        for b in protocol.BENCHMARKS:
            out['run_to_run_b0'].setdefault(m, {})[b] = {f'flip_{r}_{s}': float((b0[(b, r)] != b0[(b, s)]).mean())
                                                         for r, s in ((1, 2), (1, 3), (2, 3))}
        for v in inputs_b0.VARIANT_IDS:
            bs = inputs_b0.benchmarks(v)
            per_run, reps = {}, {}
            for r in RUNS:
                diff, point, flip = np.zeros(REPS), 0., 0.
                for b in bs:
                    ids, y, sample = data[b]
                    var, base = journal(variant_path(reused, m, v, b, r), ids), b0[(b, r)]
                    full = np.arange(len(ids))
                    fv, f0 = float(f1(y, var, full)), float(f1(y, base, full))
                    fl = float((var != base).mean())
                    out['cells'].setdefault(m, {}).setdefault(v, {}).setdefault(b, {})[str(r)] = {'f1': fv, 'f1_b0': f0, 'diff': fv-f0, 'flip': fl}
                    if r == 1:
                        old = run1['cells'][m][v][b]
                        assert abs(fv-old['f1']) < 1e-12 and abs(f0-old['f1_b0']) < 1e-12 and abs(fl-old['flip']) < 1e-12, (m, v, b)
                    point += (fv-f0)/len(bs); flip += fl/len(bs)
                    diff += (f1(y, var, sample)-f1(y, base, sample))/len(bs)
                per_run[str(r)] = {'diff': 100*point, 'flip': 100*flip, 'ci95': ci(diff)}
                reps[r] = diff
                if r == 1:
                    assert abs(per_run['1']['ci95'][0]-run1['bootstrap'][m][v]['ci95'][0]) < 1e-9 and abs(100*point-run1['bootstrap'][m][v]['diff']) < 1e-9, (m, v)
            d = [per_run[str(r)]['diff'] for r in RUNS]; fl = [per_run[str(r)]['flip'] for r in RUNS]
            out['summary'].setdefault(m, {})[v] = {'diff_mean': statistics.mean(d), 'diff_sd': statistics.stdev(d),
                                                   'flip_mean': statistics.mean(fl), 'flip_sd': statistics.stdev(fl), 'benchmarks': len(bs)}
            out['bootstrap'].setdefault(m, {})[v] = {'per_run': per_run,
                                                     'significant_in_all_runs': all(x['ci95'][0] > 0 or x['ci95'][1] < 0 for x in per_run.values()),
                                                     'run_mean': {'diff': statistics.mean(d), 'ci95': ci(sum(reps.values())/len(RUNS))}}
        print(m, {v: (round(out['summary'][m][v]['diff_mean'], 2), round(out['summary'][m][v]['diff_sd'], 2)) for v in inputs_b0.VARIANT_IDS}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
