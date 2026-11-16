"""Threshold table (tab:thresholds) over the three B0 test runs. Read-only, free.

t is tuned once on the official validation sets without the Astra training pairs (one validation run,
runs/promptval_run01_20261027), exactly as paper scripts/import_thresholds.py: one t per benchmark that maximizes its
F1 and one shared t per model that maximizes the mean F1 over the eight benchmarks, every distinct probability (and
0.5) as a candidate, ties to the t closest to 0.5. The values must reproduce sources/thresholds.json (asserted).
They are applied unchanged to B0 test Runs 1-3 of the six decision models (Run 1: runs/quality, PPLX v1.2
prompt12_run01_20261019; Runs 2/3: replications/run0{2,3}/runs/quality, PPLX v1.2 pplx12_b0_run0{2,3}_20261021); a pair is
a match if p >= t. The test-optimal t per benchmark is computed on each run (upper bound).
Per model and variant (bench, global, oracle): mean F1 over the benchmarks and its difference to t = 0.5 per run, mean
and SD (ddof=1) over the runs; mean F1 at t = 0.5 per run and over the runs (checked against sources/baseline.json,
which uses the journal predictions: the share of pairs where p >= 0.5 differs from the prediction is reported).
Paired bootstrap as scripts/import_thresholds_bootstrap.py (10,000 replicates, seed 0, the same draw sequence; the
rng is restarted per run, so every run uses the same resample of pair positions): per run (Run 1 reproduces
sources/thresholds_bootstrap.json, asserted) and run_mean (per replicate the difference averaged over the three runs).
determinism: pairs whose probability differs between runs, per model.

    .venv/bin/python src/thresholds_runs123_eval.py --paper-sources <dir with thresholds.json, thresholds_bootstrap.json,
        baseline.json> --out artifacts/results/thresholds_runs123_20261103.json
"""
import argparse
import bisect
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import protocol

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['jev', 'clef', 'clef_flash', 'luna_decisions', 'pplx_v12', 'kev']
RUNS = (1, 2, 3)
REPS = 10_000


class Sweep:
    """scripts/import_thresholds.py: F1 at any threshold on sorted probabilities."""
    def __init__(self, data):
        data = sorted(data)
        self.p = [p for p, _ in data]
        self.pos_from = [0]*(len(data)+1)
        for i in range(len(data)-1, -1, -1):
            self.pos_from[i] = self.pos_from[i+1] + data[i][1]

    def __call__(self, t):
        i = bisect.bisect_left(self.p, t)
        tp, predicted, pos = self.pos_from[i], len(self.p)-i, self.pos_from[0]
        return 2*tp/(predicted+pos) if tp else 0.


def candidates(*sets):
    return sorted({p for data in sets for p, _ in data} | {.5})


def best(score, cands):
    return max(cands, key=lambda t: (round(score(t), 12), -abs(t-.5)))


def test_path(m, b, r):
    if m == 'pplx_v12':
        ns = {1: 'runs/prompt12_run01_20261019', 2: 'runs/pplx12_b0_run02_20261021', 3: 'runs/pplx12_b0_run03_20261021'}[r]
    else:
        ns = {1: 'runs/quality', 2: 'replications/run02/runs/quality', 3: 'replications/run03/runs/quality'}[r]
    return ROOT/ns/m/'B0'/b/'test.jsonl'


def valid_pairs(m, b):
    cases = {c['pair_id']: c['label'] for c in json.loads((ROOT/'data'/b/'valid.json').read_text())}
    skip = {r['pair_id'] for r in json.loads((ROOT/'data'/b/'rules_train.json').read_text())}
    got = {r['pair_id']: r['probability_match'] for r in map(json.loads, (ROOT/'runs/promptval_run01_20261027'/m/'B0'/b/'valid.jsonl').read_text().splitlines())}
    assert got.keys() == cases.keys(), (m, b)
    return [(got[i], y) for i, y in cases.items() if i not in skip]


def f1v(y, pred, idx):
    yy, pp = y[idx], pred[idx]
    tp = ((yy == 1) & (pp == 1)).sum(axis=-1); fp = ((yy == 0) & (pp == 1)).sum(axis=-1); fn = ((yy == 1) & (pp == 0)).sum(axis=-1)
    return 2*tp/np.maximum(2*tp+fp+fn, 1)


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper-sources', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    ref = json.loads((args.paper_sources/'thresholds.json').read_text())['models']
    refboot = json.loads((args.paper_sources/'thresholds_bootstrap.json').read_text())['models']
    base = json.loads((args.paper_sources/'baseline.json').read_text())['f1']
    B = list(protocol.BENCHMARKS)
    labels = {b: {c['pair_id']: c['label'] for c in json.loads((ROOT/'data'/b/'test.json').read_text())} for b in B}
    ids = {b: list(labels[b]) for b in B}
    Y = {b: np.array([labels[b][i] for i in ids[b]], dtype=np.int8) for b in B}
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': REPS,
           'units': 'F1 as fraction in per_benchmark; means, diff, sd and ci95 in F1 points', 'models': {}, 'determinism': {}}
    P = {}
    for m in MODELS:
        val = {b: valid_pairs(m, b) for b in B}
        tb = {b: best(Sweep(val[b]), candidates(val[b])) for b in B}
        sweeps = [Sweep(val[b]) for b in B]
        tg = best(lambda t: sum(sw(t) for sw in sweeps), candidates(*val.values()))
        assert tg == ref[m]['t_global'] and all(tb[b] == ref[m]['per_benchmark'][b]['t_benchmark'] for b in B), m
        res = {'t_global': tg, 't_benchmark': tb, 'runs': {}}
        for r in RUNS:
            rows = {}
            for b in B:
                got = {x['pair_id']: (x['probability_match'], x['prediction']) for x in map(json.loads, test_path(m, b, r).read_text().splitlines())}
                assert got.keys() == set(ids[b]) and all(v[0] is not None for v in got.values()), (m, b, r)
                rows[b] = got
            P[(m, r)] = {b: np.array([rows[b][i][0] for i in ids[b]]) for b in B}
            pred = {b: np.array([rows[b][i][1] for i in ids[b]], dtype=np.int8) for b in B}
            full = {b: np.arange(len(ids[b])) for b in B}
            per = {}
            for b in B:
                p, y = P[(m, r)][b], Y[b]
                data = list(zip(p.tolist(), y.tolist()))
                st = Sweep(data)
                o = best(st, candidates(data))
                per[b] = {'f1_05': st(.5), 'f1_bench': st(tb[b]), 'f1_global': st(tg), 't_oracle': o, 'f1_oracle': st(o),
                          'f1_journal': float(f1v(y, pred[b], full[b])), 'p05_differs_from_prediction': int(((p >= .5).astype(np.int8) != pred[b]).sum())}
                if r == 1:
                    old = ref[m]['per_benchmark'][b]
                    assert abs(per[b]['f1_05']-old['test_f1_05']) < 1e-12 and abs(per[b]['f1_bench']-old['test_f1_benchmark_t']) < 1e-12 \
                        and abs(per[b]['f1_global']-old['test_f1_global_t']) < 1e-12 and abs(per[b]['f1_oracle']-old['test_f1_oracle']) < 1e-12, (m, b)
                assert abs(per[b]['f1_journal']-base[m][b][str(r)]) < 1e-12, (m, b, r)
            mean = {k: 100*statistics.mean(per[b][k] for b in B) for k in ('f1_05', 'f1_bench', 'f1_global', 'f1_oracle', 'f1_journal')}
            res['runs'][str(r)] = {'per_benchmark': per, 'mean': mean,
                                   'diff': {k: mean['f1_'+k]-mean['f1_05'] for k in ('bench', 'global', 'oracle')}}
        runs = [res['runs'][str(r)] for r in RUNS]
        res['over_runs'] = {
            'f1_05_mean': statistics.mean(x['mean']['f1_05'] for x in runs), 'f1_05_sd': sd([x['mean']['f1_05'] for x in runs]),
            'f1_journal_mean': statistics.mean(x['mean']['f1_journal'] for x in runs),
            **{f'diff_{k}_mean': statistics.mean(x['diff'][k] for x in runs) for k in ('bench', 'global', 'oracle')},
            **{f'diff_{k}_sd': sd([x['diff'][k] for x in runs]) for k in ('bench', 'global', 'oracle')},
            **{f'f1_{k}_mean': statistics.mean(x['mean']['f1_'+k] for x in runs) for k in ('bench', 'global', 'oracle')}}
        out['models'][m] = res
        out['determinism'][m] = {f'run1_vs_run{r}': int(sum((P[(m, 1)][b] != P[(m, r)][b]).sum() for b in B)) for r in (2, 3)}
        out['determinism'][m]['pairs'] = sum(len(ids[b]) for b in B)

    # bootstrap, draw sequence of import_thresholds_bootstrap.py (THRESHOLDS order there = MODELS order here)
    boot = {}
    for r in RUNS:
        rng = np.random.default_rng(0)
        for m in MODELS:
            th = out['models'][m]
            for kind in ('bench', 'global'):
                d = np.zeros(REPS); point = 0.
                for b in B:
                    p, y = P[(m, r)][b], Y[b]
                    t = th['t_benchmark'][b] if kind == 'bench' else th['t_global']
                    tuned, base05 = (p >= t).astype(np.int8), (p >= .5).astype(np.int8)
                    idx = rng.integers(0, len(y), size=(REPS, len(y)))
                    point += (f1v(y, tuned, np.arange(len(y)))-f1v(y, base05, np.arange(len(y))))/len(B)
                    d += (f1v(y, tuned, idx)-f1v(y, base05, idx))/len(B)
                boot.setdefault(m, {}).setdefault(kind, {})[r] = (point, d)
    out['bootstrap'] = {}
    for m in MODELS:
        out['bootstrap'][m] = {}
        for kind in ('bench', 'global'):
            per_run = {str(r): {'diff': 100*float(boot[m][kind][r][0]), 'ci95': [100*float(np.percentile(boot[m][kind][r][1], q)) for q in (2.5, 97.5)]}
                       for r in RUNS}
            if 1 in RUNS:
                assert abs(per_run['1']['ci95'][0]-refboot[m][kind]['ci95'][0]) < 1e-9 and abs(per_run['1']['diff']-refboot[m][kind]['diff']) < 1e-9, (m, kind)
            d = sum(boot[m][kind][r][1] for r in RUNS)/len(RUNS)
            point = sum(boot[m][kind][r][0] for r in RUNS)/len(RUNS)
            out['bootstrap'][m][kind] = {'per_run': per_run,
                                         'significant_in_all_runs': all(x['ci95'][0] > 0 or x['ci95'][1] < 0 for x in per_run.values()),
                                         'run_mean': {'diff': 100*float(point), 'ci95': [100*float(np.percentile(d, q)) for q in (2.5, 97.5)]}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
