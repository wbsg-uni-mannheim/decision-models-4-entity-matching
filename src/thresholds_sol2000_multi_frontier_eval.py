"""Multi-bootstrap of the Sol-2,000 threshold table and intervals against the mean of the frontier LLMs. Read-only, no inference
(paper writer, relayed 6 Nov).

thresholds: for all 12 cells of tab:thresholds (six decision models, t per benchmark and shared t, thresholds_sol2000_20261105.json),
tuned t minus t = 0.5, test Runs 1-3. fixed_ci95: paired bootstrap on the run mean of multibootstrap_eval.Data (10,000 replicates,
seed 0); multi_ci95: runs drawn with replacement (rng seed 1), jointly for both sides because both use the same probabilities of run
r. stored_ci95: the interval that sets the dagger in the paper (other draw sequence, thresholds_solset_eval.py).
frontier: every model of Table II minus the mean of Sol, Opus and Argon (mean over the runs and benchmarks; per replicate the mean of
the three), on all eight benchmarks and on the publication/book and product groups for PPLX, Jev, Clef, Luna Dec., Luna Gen.
Runs drawn independently per model. Point values checked against paper sources/baseline.json and sota_runs123_20261104.json.

    .venv/bin/python src/thresholds_sol2000_multi_frontier_eval.py --paper-sources <dir with baseline.json> --out artifacts/results/thresholds_sol2000_multi_frontier_20261106.json
"""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import cascade_b0_valsel_eval as cv
import protocol
from inputs_b0_eval import journal
from multibootstrap_eval import Data as BootData, RunDraws, ci, compare, side_reps
from sota_runs123_eval import later_run
from thresholds_runs123_eval import MODELS as DM6, test_path as b0_path

B = list(protocol.BENCHMARKS)
RUNS = (1, 2, 3)
SOTA = ['sol61', 'opus55', 'argon4']
TABLE2 = ['jev', 'pplx_v12', 'clef', 'clef_flash', 'kev', 'luna_decisions', 'luna', 'qwen38']
PUB = ['dblp-acm', 'dblp-scholar', 'semi-heter']
PROD = [b for b in B if b not in PUB]
RES = cv.ROOT/'artifacts/results'


def path(m, b, r):
    return cv.ROOT/f'runs/qwen38_b0_run0{r}_20261022/qwen38/B0'/b/'test.jsonl' if m == 'qwen38' else b0_path(m, b, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper-sources', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    th = json.loads((RES/'thresholds_sol2000_20261105.json').read_text())
    base = json.loads((args.paper_sources/'baseline.json').read_text())['f1']
    sota = json.loads((RES/'sota_runs123_20261104.json').read_text())['f1']
    D = BootData()
    dec1 = cv.ensemble()
    for m in DM6:
        for r in RUNS:
            for b in B:
                got = {x['pair_id']: x['probability_match'] for x in cv.lines(b0_path(m, b, r))}
                p = np.array([got[i] for i in D.ids[b]])
                D.add(('t05', m, r), b, (p >= .5).astype(np.int8))
                D.add(('t', m, 'bench', r), b, (p >= th['models'][m]['t_benchmark'][b]).astype(np.int8))
                D.add(('t', m, 'global', r), b, (p >= th['models'][m]['t_global']).astype(np.int8))
    for m in TABLE2:
        for r in RUNS:
            for b in B:
                D.add(('dm', m, r), b, journal(path(m, b, r), D.ids[b]))
    for s in SOTA:
        for r in RUNS:
            pred = {b: np.array([dec1[s][(b, i)][0] for i in D.ids[b]], dtype=np.int8) for b in B} if r == 1 else later_run(s, r, D.ids)[0]
            for b in B:
                D.add(('sota', s, r), b, pred[b])
    D.compute()
    for m in TABLE2:
        for r in RUNS:
            assert all(abs(D.point[(('dm', m, r), b)]-base[m][b][str(r)]) < 1e-12 for b in B), (m, r)
    for s in SOTA:
        for r in RUNS:
            assert all(abs(D.point[(('sota', s, r), b)]-sota[s][b][str(r)]) < 1e-12 for b in B), (s, r)
    draws = RunDraws()
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': 10_000,
           'units': 'mean F1, diff and intervals in F1 points', 'thresholds': [], 'frontier': []}
    for m in DM6:
        for kind in ('bench', 'global'):
            spec = {b: ((('dm', m), [('t', m, kind, r) for r in RUNS]), (('dm', m), [('t05', m, r) for r in RUNS])) for b in B}
            e = compare(D, draws, spec, th['bootstrap'][m][kind]['run_mean']['ci95'])
            assert abs(e['diff']-th['models'][m]['over_runs'][f'diff_{kind}_mean']) < 1e-9, (m, kind)
            out['thresholds'].append({'model': m, 'threshold': kind, 'diff': e['diff'], 'stored_ci95': e['stored_ci95'], 'stored_significant': e['stored_significant'],
                                      'fixed_ci95': e['fixed_ci95'], 'fixed_significant': e['fixed_ci95'][0] > 0 or e['fixed_ci95'][1] < 0,
                                      'multi_ci95': e['multi_ci95'], 'multi_significant': e['multi_significant'], 'pairing': e['pairing']})

    def point(kind, m, bs):
        return 100*statistics.mean(statistics.mean(D.point[((kind, m, r), b)] for r in RUNS) for b in bs)

    def reps(kind, m, bs, multi):
        return sum(side_reps(D, draws, (kind, m), [(kind, m, r) for r in RUNS], b, multi) for b in bs)/len(bs)

    def frontier(m, bs, label):
        ref = statistics.mean(point('sota', s, bs) for s in SOTA)
        res = {x: ci(reps('dm', m, bs, x)-sum(reps('sota', s, bs, x) for s in SOTA)/len(SOTA)) for x in (False, True)}
        out['frontier'].append({'model': m, 'benchmarks': label, 'mean_f1': point('dm', m, bs), 'mean_f1_frontier': ref, 'diff': point('dm', m, bs)-ref,
                                'fixed_ci95': res[False], 'multi_ci95': res[True],
                                'fixed_significant': res[False][0] > 0 or res[False][1] < 0, 'multi_significant': res[True][0] > 0 or res[True][1] < 0})

    for m in TABLE2:
        frontier(m, B, 'all eight')
    for m in ('pplx_v12', 'jev', 'clef', 'luna_decisions', 'luna'):
        frontier(m, PUB, 'publication and book')
        frontier(m, PROD, 'product')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    for x in out['thresholds']:
        print(x['model'], x['threshold'], round(x['diff'], 2), 'stored', x['stored_significant'], [round(v, 2) for v in x['stored_ci95']],
              'fixed', [round(v, 2) for v in x['fixed_ci95']], 'multi', x['multi_significant'], [round(v, 2) for v in x['multi_ci95']], flush=True)
    for x in out['frontier']:
        print(x['model'], '|', x['benchmarks'], round(x['diff'], 2), [round(v, 2) for v in x['fixed_ci95']], [round(v, 2) for v in x['multi_ci95']], flush=True)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
