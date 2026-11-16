"""Qwen3.8-27B, three local runs, minus the mean of Sol, Opus and Argon, for the Delta row of Table II (hardening session, 10 Nov). Read-only, no inference.

Input: runs/local_throughput_full_20261107/qwen_b_apisampling_r{1,2,3}.jsonl (benchmark, pair_id, label, pred; 30,225 pairs per run).
Method of the frontier block of thresholds_sol2000_multi_frontier_eval.py: mean F1 over the runs and the eight benchmarks; fixed_ci95 paired bootstrap on the run mean
(multibootstrap_eval.Data: 10,000 replicates, seed 0, pairs drawn per benchmark in protocol order, the same pairs for every arm and run); multi_ci95 also draws the three
runs with replacement (rng seed 1), independently per model; per replicate the frontier value is the mean of the three LLMs.

    .venv/bin/python src/qwen38_local_vs_frontier_eval.py --out artifacts/results/qwen38_local_vs_frontier_20261110.json
"""
import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import cascade_b0_valsel_eval as cv
from multibootstrap_eval import B, REPS, RES, ROOT, Data, RunDraws, ci, sig, side_reps
from sota_runs123_eval import later_run

RUNS = (1, 2, 3)
SOTA = ['sol61', 'opus55', 'argon4']
DIR = ROOT/'runs/local_throughput_full_20261107'
SOTA_FILE = RES/'sota_runs123_20261104.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    D = Data()
    ids, y = D.ids, D.y
    inputs = {SOTA_FILE.name: hashlib.sha256(SOTA_FILE.read_bytes()).hexdigest()}
    per_run = {}
    for r in RUNS:
        p = DIR/f'qwen_b_apisampling_r{r}.jsonl'
        inputs[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        got = {}
        for x in map(json.loads, p.read_text().splitlines()):
            assert (x['benchmark'], x['pair_id']) not in got, x
            got[(x['benchmark'], x['pair_id'])] = x
        assert len(got) == sum(len(v) for v in ids.values()), (r, len(got))
        for b in B:
            assert all(got[(b, i)]['label'] == int(l) for i, l in zip(ids[b], y[b])), (r, b)
            assert all(got[(b, i)]['pred'] in (0, 1) for i in ids[b]), (r, b)
            D.add(('dm', 'qwen38', r), b, np.array([got[(b, i)]['pred'] for i in ids[b]], dtype=np.int8))
    dec1 = cv.ensemble()
    for s in SOTA:
        for r in RUNS:
            pred = {b: np.array([dec1[s][(b, i)][0] for i in ids[b]], dtype=np.int8) for b in B} if r == 1 else later_run(s, r, ids)[0]
            for b in B:
                D.add(('sota', s, r), b, pred[b])
    D.compute()
    sota = json.loads(SOTA_FILE.read_text())['f1']
    for s in SOTA:
        for r in RUNS:
            assert all(abs(D.point[(('sota', s, r), b)]-sota[s][b][str(r)]) < 1e-12 for b in B), (s, r)
    draws = RunDraws()

    def point(kind, m):
        return 100*statistics.mean(statistics.mean(D.point[((kind, m, r), b)] for r in RUNS) for b in B)

    def reps(kind, m, multi):
        return sum(side_reps(D, draws, (kind, m), [(kind, m, r) for r in RUNS], b, multi) for b in B)/len(B)

    q, ref = point('dm', 'qwen38'), statistics.mean(point('sota', s) for s in SOTA)
    fixed, multi = (ci(reps('dm', 'qwen38', x)-sum(reps('sota', s, x) for s in SOTA)/len(SOTA)) for x in (False, True))
    for r in RUNS:
        per_run[str(r)] = {'mean_f1': 100*statistics.mean(D.point[(('dm', 'qwen38', r), b)] for b in B),
                           'per_benchmark': {b: 100*D.point[(('dm', 'qwen38', r), b)] for b in B}}
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': REPS, 'seed': 0, 'run_draw_seed': 1, 'inputs_sha256': inputs,
           'units': 'F1, difference and intervals in points', 'frontier': [{
               'model': 'qwen38 (local runs)', 'benchmarks': 'all eight', 'mean_f1': q, 'mean_f1_sd_over_runs': statistics.stdev(x['mean_f1'] for x in per_run.values()),
               'mean_f1_frontier': ref, 'diff': q-ref, 'fixed_ci95': fixed, 'multi_ci95': multi, 'fixed_significant': sig(fixed), 'multi_significant': sig(multi),
               'significant_both': sig(fixed) and sig(multi)}], 'per_run': per_run,
           'per_benchmark_mean': {b: statistics.mean(100*D.point[(('dm', 'qwen38', r), b)] for r in RUNS) for b in B}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1, default=lambda o: o.item())
    print(json.dumps(out['frontier'], indent=1), {r: round(x['mean_f1'], 2) for r, x in per_run.items()}, flush=True)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
