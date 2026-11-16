"""Prompt study (Table VIII) and validation selection (Table IX) over three test runs. Read-only, free.

Runs: the ten zero-shot prompts (B0 + nine; without Astra and handwritten rules), seven models, eight full core
test sets. Run 1: prompt12_run01_20261019 (B0 of the six base models from runs/quality, PPLX v1.2 its own B0).
Runs 2 and 3: prompt12_run0{2,3}_20261103 for the nine prompts (Aaron, 3 Nov), B0 from the baseline replications
(replications/run0{2,3}/runs/quality; PPLX v1.2 pplx12_b0_run0{2,3}_20261021). F1 from the journal predictions.
cells: F1 per model, prompt and benchmark for each run, mean and sample SD (ddof=1) over the three runs.
table8: per model and prompt the F1 averaged over the benchmarks, per run and as the 3-run mean (mean of the run
  averages); SD over the ten prompts; best prompt (highest average rounded to 2 decimals, ties to B0, as
  render_prompts.py) and its difference to B0.
table9: the prompt selected per model and benchmark on the official validation sets (unchanged, one validation run,
  paper sources/prompt_validation.json), its test F1 per run and as the 3-run mean; difference to B0; best prompt per
  benchmark on test (oracle) on the 3-run means; how often B0 is selected.
Bootstrap (paired, stratified per benchmark, 10,000 replicates, seed 0; the same pair resample for every prompt and
run): best zero-shot prompt vs B0 with the resample of import_prompt_bootstrap.py (paper benchmark order) and
validation selection vs B0 with the resample of promptval_bootstrap_eval.py (protocol order), so that Run 1
reproduces sources/prompt_bootstrap.json (best_zero_shot_vs_b0) and sources/prompt_validation_bootstrap.json
(valid_selection). Variants:
  per_run: run k against B0 run k (the best prompt chosen on that run's table);
  run_mean: the 3-run means, per replicate the F1 of each prompt averaged over the three runs (pairs resampled, runs
  fixed); plus the SD over runs of the per-run point difference.
No inference.

    .venv/bin/python src/prompt12_runs123_eval.py --paper-sources <dir with prompt_bootstrap.json, prompt_validation.json,
        prompt_validation_bootstrap.json> --out artifacts/results/prompt12_runs123_20261103.json
"""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import protocol

ROOT = Path(__file__).resolve().parents[1]
PAPER_ORDER = ['abt-buy', 'walmart-amazon', 'wdc', 'dblp-acm', 'dblp-scholar', 'dn7-walmart-amazon', 'billiger-de', 'semi-heter']
MODELS = ['jev', 'clef', 'clef_flash', 'luna_decisions', 'pplx_v12', 'kev', 'luna']
ZERO_SHOT = ['B0', 'PE_general_simple', 'PE_general_complex', 'PE_domain_simple', 'PE_domain_complex', 'NA_simple',
             'NA_complex', 'X_P_domain_0', 'X_P_conservative_0', 'X_P_evidence_0']
RUNS = (1, 2, 3)  # --runs 1 for the Run-1 reproduction check only
REPS = 10_000
SELECTION = ROOT/'configs/inputs_prompt_selection_20261019.json'


def path(m, p, b, r):
    if p == 'B0':
        if m == 'pplx_v12':
            ns = {1: 'runs/prompt12_run01_20261019', 2: 'runs/pplx12_b0_run02_20261021', 3: 'runs/pplx12_b0_run03_20261021'}[r]
        else:
            ns = {1: 'runs/quality', 2: 'replications/run02/runs/quality', 3: 'replications/run03/runs/quality'}[r]
        return ROOT/ns/m/'B0'/b/'test.jsonl'
    return ROOT/'runs'/('prompt12_run01_20261019' if r == 1 else f'prompt12_run0{r}_20261103')/m/p/b/'test.jsonl'


def sd(values):
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else None


def f1(tp, fp, fn):
    return 2*tp/np.maximum(2*tp+fp+fn, 1)


def resample_counts(order):
    """Per benchmark the count matrix W (REPS x n) of the resample rng.integers(0, n, (REPS, n)), seed 0, in this order."""
    rng = np.random.default_rng(0)
    out = {}
    for b in order:
        n = len(Y[b])
        idx = rng.integers(0, n, size=(REPS, n))
        w = np.zeros((REPS, n), dtype=np.float32)
        for i in range(REPS):
            w[i] = np.bincount(idx[i], minlength=n)
        out[b] = w
    return out


def replicate_f1(w, b):
    """F1 per replicate for every (model, prompt, run): {key: array(REPS)}."""
    keys = [(m, p, r) for m in MODELS for p in ZERO_SHOT for r in RUNS]
    y = Y[b].astype(np.float32)
    cols = []
    for k in keys:
        p = PRED[k][b].astype(np.float32)
        cols += [y*p, (1-y)*p, y*(1-p)]
    c = (w @ np.stack(cols, axis=1)).astype(np.float64)  # counts are exact in float32
    return {k: f1(c[:, 3*i], c[:, 3*i+1], c[:, 3*i+2]) for i, k in enumerate(keys)}


def interval(point, rep):
    return {'diff': 100*point, 'ci95': [100*float(np.percentile(rep, 2.5)), 100*float(np.percentile(rep, 97.5))],
            'share_le_0': float((rep <= 0).mean())}


def main():
    global Y, PRED
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper-sources', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--runs', type=int, nargs='+', default=[1, 2, 3])
    args = ap.parse_args()
    global RUNS
    RUNS = tuple(args.runs)
    assert not args.out.exists(), args.out
    boot1 = json.loads((args.paper_sources/'prompt_bootstrap.json').read_text())['models']
    pval = json.loads((args.paper_sources/'prompt_validation.json').read_text())
    vboot1 = json.loads((args.paper_sources/'prompt_validation_bootstrap.json').read_text())['models']
    sel_cfg = json.loads(SELECTION.read_text())['f1']

    Y, PRED, ids = {}, {}, {}
    for b in PAPER_ORDER:
        cases = json.loads((ROOT/'data'/b/'test.json').read_text())
        ids[b] = [c['pair_id'] for c in cases]
        Y[b] = np.array([c['label'] for c in cases], dtype=np.int8)
    F1 = {}
    for m in MODELS:
        for p in ZERO_SHOT:
            for r in RUNS:
                PRED[(m, p, r)] = {}
                for b in PAPER_ORDER:
                    rows = {x['pair_id']: x['prediction'] for x in map(json.loads, path(m, p, b, r).read_text().splitlines())}
                    assert rows.keys() == set(ids[b]) and all(v in (0, 1) for v in rows.values()), (m, p, b, r)
                    pr = np.array([rows[i] for i in ids[b]], dtype=np.int8)
                    PRED[(m, p, r)][b] = pr
                    tp = int(((Y[b] == 1) & (pr == 1)).sum()); fp = int(((Y[b] == 0) & (pr == 1)).sum()); fn = int(((Y[b] == 1) & (pr == 0)).sum())
                    F1.setdefault(m, {}).setdefault(p, {}).setdefault(b, {})[r] = float(f1(tp, fp, fn))
                    if r == 1:
                        assert abs(F1[m][p][b][1]-sel_cfg[m][p][b]) < 1e-12, (m, p, b)
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': REPS,
           'units': 'F1 as fraction; table means, SD over prompts, diff and ci95 in F1 points', 'prompts': ZERO_SHOT,
           'cells': {m: {p: {b: {'runs': [F1[m][p][b][r] for r in RUNS], 'mean': statistics.mean(F1[m][p][b].values()),
                                 'sd': sd(F1[m][p][b].values())} for b in PAPER_ORDER} for p in ZERO_SHOT} for m in MODELS}}

    def avg(m, p, r=None):
        return 100*statistics.mean(F1[m][p][b][r] if r else statistics.mean(F1[m][p][b].values()) for b in PAPER_ORDER)

    def best_of(m, r=None):
        return max(ZERO_SHOT, key=lambda p: (round(avg(m, p, r), 2), p == 'B0'))
    t8 = {}
    for m in MODELS:
        x = {'mean_f1': {p: avg(m, p) for p in ZERO_SHOT}, 'per_run_mean_f1': {str(r): {p: avg(m, p, r) for p in ZERO_SHOT} for r in RUNS}}
        x['sd_over_prompts'] = sd(x['mean_f1'].values())
        x['sd_over_prompts_per_run'] = {str(r): sd(x['per_run_mean_f1'][str(r)].values()) for r in RUNS}
        x['best'] = best_of(m); x['best_per_run'] = {str(r): best_of(m, r) for r in RUNS}
        x['best_minus_b0'] = round(x['mean_f1'][x['best']], 2)-round(x['mean_f1']['B0'], 2)
        x['sd_over_runs_of_prompt_mean'] = {p: sd(x['per_run_mean_f1'][str(r)][p] for r in RUNS) for p in ZERO_SHOT}
        t8[m] = x
    out['table8'] = t8

    sel = {m: {b: pval['selection'][m][b]['prompt'] for b in PAPER_ORDER} for m in MODELS}
    t9 = {}
    for m in MODELS:
        per_run = {str(r): 100*statistics.mean(F1[m][sel[m][b]][b][r] for b in PAPER_ORDER) for r in RUNS}
        mean = statistics.mean(per_run.values())
        assert abs(per_run['1']/100-pval['mean_test_f1_per_model'][m]['test_f1']) < 1e-12, m
        oracle = 100*statistics.mean(max(statistics.mean(F1[m][q][b].values()) for q in ZERO_SHOT) for b in PAPER_ORDER)
        t9[m] = {'selection': sel[m], 'test_f1_per_run': per_run, 'test_f1_mean': mean, 'b0_mean': t8[m]['mean_f1']['B0'],
                 'minus_b0': round(mean, 2)-round(t8[m]['mean_f1']['B0'], 2),
                 'oracle_mean': oracle, 'oracle_minus_b0': round(oracle, 2)-round(t8[m]['mean_f1']['B0'], 2),
                 'b0_selected': sum(sel[m][b] == 'B0' for b in PAPER_ORDER),
                 'per_benchmark': {b: {'prompt': sel[m][b], 'test_f1_runs': [F1[m][sel[m][b]][b][r] for r in RUNS],
                                       'test_f1_mean': statistics.mean(F1[m][sel[m][b]][b].values()),
                                       'b0_mean': statistics.mean(F1[m]['B0'][b].values())} for b in PAPER_ORDER}}
    out['table9'] = t9

    # best zero-shot prompt vs B0 (resample of import_prompt_bootstrap.py)
    W = resample_counts(PAPER_ORDER)
    rep = {b: replicate_f1(W[b], b) for b in PAPER_ORDER}
    del W
    bz = {}
    for m in MODELS:
        res = {'per_run': {}}
        for r in RUNS:
            best = t8[m]['best_per_run'][str(r)]
            if best == 'B0':
                res['per_run'][str(r)] = {'best': 'B0'}
                continue
            point = statistics.mean(F1[m][best][b][r]-F1[m]['B0'][b][r] for b in PAPER_ORDER)
            d = sum(rep[b][(m, best, r)]-rep[b][(m, 'B0', r)] for b in PAPER_ORDER)/len(PAPER_ORDER)
            res['per_run'][str(r)] = {'best': best, **interval(point, d)}
        r1 = res['per_run']['1']
        if 'best_zero_shot_vs_b0' in boot1[m]:
            assert r1['best'] == boot1[m]['best_zero_shot_vs_b0']['best'] and abs(r1['ci95'][0]-boot1[m]['best_zero_shot_vs_b0']['ci95'][0]) < 1e-9, m
        best = t8[m]['best']
        if best != 'B0':
            point = statistics.mean(statistics.mean(F1[m][best][b].values())-statistics.mean(F1[m]['B0'][b].values()) for b in PAPER_ORDER)
            d = sum(sum(rep[b][(m, best, r)]-rep[b][(m, 'B0', r)] for r in RUNS)/len(RUNS) for b in PAPER_ORDER)/len(PAPER_ORDER)
            per = [statistics.mean(F1[m][best][b][r]-F1[m]['B0'][b][r] for b in PAPER_ORDER) for r in RUNS]
            res['run_mean'] = {'best': best, **interval(point, d), 'per_run_point_diff': [100*x for x in per],
                               'sd_over_runs': 100*sd(per) if len(per) > 1 else None}
        else:
            res['run_mean'] = {'best': 'B0'}
        bz[m] = res
    del rep

    # validation selection vs B0 (resample of promptval_bootstrap_eval.py)
    W = resample_counts(list(protocol.BENCHMARKS))
    rep = {b: replicate_f1(W[b], b) for b in protocol.BENCHMARKS}
    del W
    vb = {}
    for m in MODELS:
        res = {'per_run': {}}
        for r in RUNS:
            point = statistics.mean(F1[m][sel[m][b]][b][r]-F1[m]['B0'][b][r] for b in PAPER_ORDER)
            d = sum(rep[b][(m, sel[m][b], r)]-rep[b][(m, 'B0', r)] for b in protocol.BENCHMARKS)/len(PAPER_ORDER)
            res['per_run'][str(r)] = interval(point, d)
        assert abs(res['per_run']['1']['ci95'][0]-vboot1[m]['valid_selection']['ci95'][0]) < 1e-9 \
               and abs(res['per_run']['1']['diff']-vboot1[m]['valid_selection']['diff']) < 1e-9, m
        point = statistics.mean(statistics.mean(F1[m][sel[m][b]][b].values())-statistics.mean(F1[m]['B0'][b].values()) for b in PAPER_ORDER)
        d = sum(sum(rep[b][(m, sel[m][b], r)]-rep[b][(m, 'B0', r)] for r in RUNS)/len(RUNS) for b in protocol.BENCHMARKS)/len(PAPER_ORDER)
        per = [res['per_run'][str(r)]['diff'] for r in RUNS]
        res['run_mean'] = {**interval(point, d), 'sd_over_runs': sd(per)}
        vb[m] = res
    out['bootstrap'] = {'best_zero_shot_vs_b0': bz, 'valid_selection_vs_b0': vb,
                        'checks': 'Run 1 reproduces prompt_bootstrap.json best_zero_shot_vs_b0 and prompt_validation_bootstrap.json valid_selection (asserted).'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
