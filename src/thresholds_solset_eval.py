"""Threshold table (tab:thresholds) with t tuned on our Sol labels instead of the official validation labels. Read-only.

Aaron (5 Nov, relayed): Section 3.3 moves to the labelled-data chapter; t is tuned on the Sol-labelled 200 and 2,000 sets
(all pairs of each set except those whose records equal a test pair, 26 in the 200 and 96 in the 2,000 set, as
valsel.py drops such validation pairs;
configs/solset_b0_cases_20261105.json; B0 probabilities from runs/solset_b0_run01_20261105, one
run) instead of the official validation sets. Tuning rule of thresholds_runs123_eval.py (scripts/import_thresholds.py):
one t per benchmark that maximizes its F1 against the Sol labels and one shared t per model that maximizes the mean F1
over the eight benchmarks, every distinct probability (and 0.5) as a candidate, ties to the t closest to 0.5.
The t are applied unchanged to B0 test Runs 1-3 of the six decision models (thresholds_runs123_eval.test_path); a pair is
a match if p >= t. Reference: t tuned on the official validation sets (recomputed, asserted against
sources/thresholds.json), as in the current table.
Per model, tuning source (valid, sol200, sol2000) and variant (bench, global): t, F1 on the tuning set at t and at 0.5,
test F1 per benchmark and run, mean F1 and its difference to t = 0.5 per run, mean and SD (ddof=1) over the runs.
Paired bootstrap with the draw sequence of thresholds_runs123_eval.py (10,000 replicates, rng seed 0 restarted per run,
models in MODELS order, bench before global, benchmarks in protocol order; every source uses the same resample):
each source against t = 0.5 and sol200 / sol2000 against valid, per run and run_mean (per replicate the difference
averaged over the three runs). valid against 0.5 reproduces sources/thresholds_runs123.json (asserted).
Besides --out, one file per Sol set (thresholds_sol200_*.json, thresholds_sol2000_*.json) in the layout of
thresholds_runs123_eval.py (t, runs with f1_05/bench/global/oracle, over_runs, bootstrap against 0.5), for the paper's
import_thresholds_runs123.py; bootstrap_vs_valid holds the comparison with the validation-tuned t.

    .venv/bin/python src/thresholds_solset_eval.py --paper-sources <dir with thresholds.json, thresholds_runs123.json> \
        --out artifacts/results/thresholds_solset_20261105.json
"""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import protocol
import solset_b0
from thresholds_runs123_eval import MODELS, REPS, RUNS, Sweep, best, candidates, sd, test_path, valid_pairs

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'runs/solset_b0_run01_20261105'
SOURCES = ('valid', 'sol200', 'sol2000')
KINDS = ('bench', 'global')
B = list(protocol.BENCHMARKS)


TEST_INPUTS = {}


def test_inputs(b):
    if b not in TEST_INPUTS:
        TEST_INPUTS[b] = {c['input_hash'] for c in json.loads((ROOT/'data'/b/'test.json').read_text())}
    return TEST_INPUTS[b]


def sol_pairs(m, b, ids):
    """(p, Sol label) of a set; pairs whose records equal a test pair (same input_hash, other ids) are left out."""
    rows = json.loads((ROOT/'data'/b/'solset.json').read_text())
    cases = {c['pair_id']: c['label'] for c in rows}
    skip = {c['pair_id'] for c in rows if c['input_hash'] in test_inputs(b)}
    got = {}
    for x in map(json.loads, (ROOT/NAMESPACE/m/'B0'/b/'solset.jsonl').read_text().splitlines()):
        assert x['pair_id'] not in got and x['probability_match'] is not None, (m, b, x['pair_id'])
        got[x['pair_id']] = x['probability_match']
    assert got.keys() == cases.keys(), (m, b)
    return [(got[i], cases[i]) for i in ids if i not in skip]


def tune(data):
    """{bench: {b: t}, global: t} and the tuning F1 at t and at 0.5."""
    sweeps = {b: Sweep(data[b]) for b in B}
    tb = {b: best(sweeps[b], candidates(data[b])) for b in B}
    tg = best(lambda t: sum(sw(t) for sw in sweeps.values()), candidates(*data.values()))
    return {'t_benchmark': tb, 't_global': tg,
            'tuning': {b: {'n': len(data[b]), 'positive': sum(y for _, y in data[b]), 'f1_05': sweeps[b](.5),
                           'f1_bench': sweeps[b](tb[b]), 'f1_global': sweeps[b](tg)} for b in B}}


def f1_rows(yy, pp):
    tp = (yy & pp).sum(axis=-1); fp = ((1-yy) & pp).sum(axis=-1); fn = (yy & (1-pp)).sum(axis=-1)
    return 2*tp/np.maximum(2*tp+fp+fn, 1)


def ci(x):
    return [100*float(np.percentile(x, q)) for q in (2.5, 97.5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper-sources', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true', help='before the Sol-set run: sol sources reuse the valid t, asserts only, no output')
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    ref = json.loads((args.paper_sources/'thresholds.json').read_text())['models']
    ref3 = json.loads((args.paper_sources/'thresholds_runs123.json').read_text())
    sets = solset_b0.tuning_sets(ROOT)
    cases_cfg = json.loads((ROOT/solset_b0.CASES).read_text())['benchmarks']
    labels = {b: {c['pair_id']: c['label'] for c in json.loads((ROOT/'data'/b/'test.json').read_text())} for b in B}
    ids = {b: list(labels[b]) for b in B}
    Y = {b: np.array([labels[b][i] for i in ids[b]], dtype=np.int8) for b in B}
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': REPS,
           'units': 'F1 as fraction in per_benchmark and tuning; means, diff, sd and ci95 in F1 points',
           'tuning_sets': {n: {b: {k: cases_cfg[b]['sets'][n][k] for k in ('n', 'positive')} for b in B} for n in solset_b0.SIZES},
           'overlap_with_official_splits': {b: cases_cfg[b]['overlap'] for b in B},
           'left_out_test_input': {n: {b: len(set(sets[n][b]) & {c['pair_id'] for c in json.loads((ROOT/'data'/b/'solset.json').read_text()) if c['input_hash'] in test_inputs(b)}) for b in B} for n in solset_b0.SIZES},
           'models': {}}
    P = {}
    for m in MODELS:
        th = {'valid': tune({b: valid_pairs(m, b) for b in B})}
        th.update({'sol'+n: th['valid'] if args.smoke else tune({b: sol_pairs(m, b, sets[n][b]) for b in B}) for n in solset_b0.SIZES})
        v = th['valid']
        assert v['t_global'] == ref[m]['t_global'] and all(v['t_benchmark'][b] == ref[m]['per_benchmark'][b]['t_benchmark'] for b in B), m
        res = {s: {**th[s], 'runs': {}} for s in SOURCES}
        for r in RUNS:
            rows = {}
            for b in B:
                got = {x['pair_id']: x['probability_match'] for x in map(json.loads, test_path(m, b, r).read_text().splitlines())}
                assert got.keys() == set(ids[b]) and all(p is not None for p in got.values()), (m, b, r)
                rows[b] = np.array([got[i] for i in ids[b]])
            P[(m, r)] = rows
            sweeps = {b: Sweep(list(zip(rows[b].tolist(), Y[b].tolist()))) for b in B}
            f05 = 100*statistics.mean(sweeps[b](.5) for b in B)
            oracle = {b: best(sweeps[b], candidates(list(zip(rows[b].tolist(), Y[b].tolist())))) for b in B}
            for s in SOURCES:
                per = {b: {'f1_05': sweeps[b](.5), 'f1_bench': sweeps[b](th[s]['t_benchmark'][b]), 'f1_global': sweeps[b](th[s]['t_global']),
                           't_oracle': oracle[b], 'f1_oracle': sweeps[b](oracle[b])} for b in B}
                mean = {k: 100*statistics.mean(per[b][k] for b in B) for k in ('f1_05', 'f1_bench', 'f1_global', 'f1_oracle')}
                res[s]['runs'][str(r)] = {'per_benchmark': per, 'mean': mean, 'diff': {k: mean['f1_'+k]-f05 for k in KINDS+('oracle',)}}
        for s in SOURCES:
            runs = [res[s]['runs'][str(r)] for r in RUNS]
            res[s]['over_runs'] = {'f1_05_mean': statistics.mean(x['mean']['f1_05'] for x in runs), 'f1_05_sd': sd([x['mean']['f1_05'] for x in runs]),
                                   **{f'f1_{k}_mean': statistics.mean(x['mean']['f1_'+k] for x in runs) for k in KINDS+('oracle',)},
                                   **{f'diff_{k}_mean': statistics.mean(x['diff'][k] for x in runs) for k in KINDS+('oracle',)},
                                   **{f'diff_{k}_sd': sd([x['diff'][k] for x in runs]) for k in KINDS+('oracle',)}}
        for k in ('f1_05_mean', 'f1_05_sd', 'f1_oracle_mean')+tuple(f'diff_{k}_{x}' for k in KINDS+('oracle',) for x in ('mean', 'sd')):
            assert abs(res['valid']['over_runs'][k]-ref3['models'][m]['over_runs'][k]) < 1e-9, (m, k)  # the current table
        for r in RUNS:
            for b in B:
                assert all(abs(res['valid']['runs'][str(r)]['per_benchmark'][b][k]-ref3['models'][m]['runs'][str(r)][b][k]) < 1e-12
                           for k in ('f1_05', 'f1_bench', 'f1_global', 'f1_oracle')), (m, r, b)
        out['models'][m] = res
        print(m, {s: (res[s]['t_global'], round(res[s]['over_runs']['diff_bench_mean'], 2), round(res[s]['over_runs']['diff_global_mean'], 2))
                  for s in SOURCES}, flush=True)

    # bootstrap: draw sequence of thresholds_runs123_eval.py; every source and 0.5 on the same resample
    comps = [(s, '05') for s in SOURCES] + [('sol200', 'valid'), ('sol2000', 'valid')]
    boot = {}
    for r in RUNS:
        rng = np.random.default_rng(0)
        for m in MODELS:
            th = out['models'][m]
            for kind in KINDS:
                d = {c: np.zeros(REPS) for c in comps}; point = {c: 0. for c in comps}
                for b in B:
                    p, y = P[(m, r)][b], Y[b]
                    idx = rng.integers(0, len(y), size=(REPS, len(y)))
                    yy = y[idx]
                    preds = {'05': (p >= .5).astype(np.int8)}
                    for s in SOURCES:
                        t = th[s]['t_benchmark'][b] if kind == 'bench' else th[s]['t_global']
                        preds[s] = (p >= t).astype(np.int8)
                    f_rep = {s: f1_rows(yy, pr[idx]) for s, pr in preds.items()}
                    f_pt = {s: float(f1_rows(y, pr)) for s, pr in preds.items()}
                    for a, c in comps:
                        d[(a, c)] += (f_rep[a]-f_rep[c])/len(B); point[(a, c)] += (f_pt[a]-f_pt[c])/len(B)
                for c in comps:
                    boot.setdefault(m, {}).setdefault(kind, {}).setdefault(c, {})[r] = (point[c], d[c])
    out['bootstrap'] = {}
    for m in MODELS:
        for kind in KINDS:
            for a, c in comps:
                x = boot[m][kind][(a, c)]
                per_run = {str(r): {'diff': 100*x[r][0], 'ci95': ci(x[r][1])} for r in RUNS}
                mean_d = sum(x[r][1] for r in RUNS)/len(RUNS)
                entry = {'per_run': per_run, 'run_mean': {'diff': 100*sum(x[r][0] for r in RUNS)/len(RUNS), 'ci95': ci(mean_d)},
                         'significant_in_all_runs': all(v['ci95'][0] > 0 or v['ci95'][1] < 0 for v in per_run.values())}
                lo, hi = entry['run_mean']['ci95']
                entry['run_mean']['significant'] = lo > 0 or hi < 0
                if (a, c) == ('valid', '05'):
                    old = ref3['bootstrap'][m][kind]
                    assert abs(entry['run_mean']['diff']-old['run_mean']['diff']) < 1e-9 \
                        and max(abs(u-w) for u, w in zip(entry['run_mean']['ci95'], old['run_mean']['ci95'])) < 1e-9, (m, kind)
                out['bootstrap'].setdefault(m, {}).setdefault(kind, {})[f'{a}_vs_{c}'] = entry
    if args.smoke:
        print('smoke ok: valid source reproduces thresholds.json and thresholds_runs123.json')
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    print('wrote', args.out)
    for n in solset_b0.SIZES:  # one file per Sol set in the layout of thresholds_runs123_eval.py (paper import_thresholds_runs123.py)
        s = 'sol'+n
        layout = {'description': f'Layout of thresholds_runs123_eval.py; t tuned on the Sol-labelled {n} set. Extract of {args.out.name}.\n'+__doc__,
                  'created_at': out['created_at'], 'replicates': REPS, 'units': out['units'], 'tuning_source': s,
                  'models': {m: {k: out['models'][m][s][k] for k in ('t_global', 't_benchmark', 'tuning', 'runs', 'over_runs')} for m in MODELS},
                  'bootstrap': {m: {kind: out['bootstrap'][m][kind][f'{s}_vs_05'] for kind in KINDS} for m in MODELS},
                  'bootstrap_vs_valid': {m: {kind: out['bootstrap'][m][kind][f'{s}_vs_valid'] for kind in KINDS} for m in MODELS},
                  'determinism': ref3['determinism']}
        path = args.out.with_name(args.out.stem.replace('thresholds_solset', f'thresholds_sol{n}')+'.json')
        with path.open('x') as f:
            json.dump(layout, f, indent=1)
        print('wrote', path)


if __name__ == '__main__':
    main()
