"""Per-cell significance for Tables V, VI, VII and IX of the paper (Chris, relayed 10 Nov). Read-only, no inference.

Method of Section 2.4: paired bootstrap, 10,000 replicates, seed 0, per benchmark the test pairs drawn with replacement (the same pairs for
both configurations and all runs), difference of the three-run means (fixed_ci95); multi-bootstrap with the three runs drawn with
replacement (rng seed 1, multi_ci95). significant: both 95 % intervals exclude zero. Run draws are independent when the two sides are
separate API calls, joint when both sides come from the same probabilities of run r (cascades vs the model alone).
tab:prompt-selection: per benchmark and model, the prompt selected on validation vs DP (prompt12_runs123_20261103.json selection); cells where DP
was selected and the row 'Delta Best on test - DP' (selection on test, upper bound) are not_tested.
tab:inputs: the two rows 'Mean over variants': per replicate the mean over the variants of the group of (variant minus DP, mean over that variant's
benchmarks), so Lowercase enters with its own four benchmarks as one variant, exactly as the table averages.
tab:cascade: per benchmark and in the mean, + Sol and + ensemble at t = 0.05 vs the same model alone.
tab:labeled: per benchmark and in the mean, rows of a decision model vs DP of that model; Sol, Ditto (seeds as runs) and the PPLX cascades vs PPLX v1.2 DP.
Benchmark means of the labeled rows are checked against later_tables_sol2000_runs123_20261105_v2.json, the cascade means against cascade_fixed_t_20261107.json.

    .venv/bin/python src/table_cell_significance_eval.py --out artifacts/results/table_cell_significance_20261110.json
"""
import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import cascade_b0_valsel_eval as cv
import cascade_solset_runs123_eval as cs
import inputs_b0
from inputs_b0_eval import journal
from inputs_b0_runs123_eval import b0_path, variant_path
from multibootstrap_eval import LC, P_MODELS, SEEDS, SIZES, Data, RunDraws, ci, compare, side_reps
from prompt12_runs123_eval import path as prompt_path

B, RUNS, RES, ROOT = cs.B, cs.RUNS, cs.RES, cs.ROOT
T = .05
CONFIGS = ('sol61', 'majority')
GROUPS = {'Serialization': ['F1_values', 'F2_json', 'F3_aligned', 'F4_no_missing'],
          'Perturbation': ['R1_swap', 'R2_attr_reverse', 'R3_lowercase', 'R4_attr_del', 'R6_typos']}
INPUT_FILES = ['prompt12_runs123_20261103.json', 'inputs_b0_runs123_20261104.json', 'cascade_fixed_t_20261107.json',
               'cascade_sol2000_runs123_20261105_v2.json', 'later_tables_sol2000_runs123_20261105_v2.json', 'sota_runs123_20261104.json']


def excl(c):
    return c[0] > 0 or c[1] < 0


def tested(diff, fixed, multi, pairing):
    return {'status': 'tested', 'diff': diff, 'fixed_ci95': fixed, 'multi_ci95': multi, 'fixed_excludes_zero': excl(fixed), 'multi_excludes_zero': excl(multi),
            'significant': excl(fixed) and excl(multi), 'bootstraps_disagree': excl(fixed) != excl(multi), 'pairing': pairing}


def not_tested(reason):
    return {'status': 'not_tested', 'reason': reason}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    pr = json.loads((RES/INPUT_FILES[0]).read_text())['table9']
    inp = json.loads((RES/INPUT_FILES[1]).read_text())['summary']
    fixed_t = {(x['model'], x['partner']): x for x in json.loads((RES/INPUT_FILES[2]).read_text())['cells'] if x['t'] == T}
    tuned_t = json.loads((RES/INPUT_FILES[3]).read_text())['table9']['pplx_v12']['t']
    lt = {(s['model'], s['row']): s for s in json.loads((RES/INPUT_FILES[4]).read_text())['significance'] if s['table'] == 'table10'}
    D = Data()
    ids, y = D.ids, D.y
    for m in P_MODELS:
        for r in RUNS:
            for b in B:
                D.add(('B0', m, r), b, journal(b0_path(m, b, r), ids[b]))
        for p in {pr[m]['selection'][b] for b in B}-{'B0'}:
            for r in RUNS:
                for b in B:
                    D.add(('prompt', m, p, r), b, journal(prompt_path(m, p, b, r), ids[b]))
    inputs_b0.register_models()
    reused = inputs_b0.reused(ROOT)
    for m in inputs_b0.RUN_MODELS:
        for vs in GROUPS.values():
            for v in vs:
                for r in RUNS:
                    for b in inputs_b0.benchmarks(v):
                        D.add(('input', m, v, r), b, journal(variant_path(reused, m, v, b, r), ids[b]))
    dec1 = cv.ensemble()
    partner = {r: cs.partner_run(r, ids, dec1)[0] for r in RUNS}
    for m in cs.DM4:
        for r in RUNS:
            test = cs.test_rows(lambda b: cs.b0_path(m, b, r), ids)
            _, preds = cs.cascade(y, test, partner[r], T)
            for b in B:
                D.add(('alone', m, r), b, preds['alone'][b])
                for c in CONFIGS:
                    D.add(('casc', m, c, r), b, preds[c][b])
            if m == 'pplx_v12':
                _, tp = cs.cascade(y, test, partner[r], tuned_t)
                for c in CONFIGS:
                    for b in B:
                        D.add(('tuned', c, r), b, tp[c][b])
    for r in RUNS:
        for b in B:
            D.add(('sol', r), b, partner[r]['sol61'][b][0])
    for s in SIZES:
        for seed in SEEDS:
            for b in B:
                summ = json.loads((LC/f'ditto/output/{b}_set{s}_seed{seed}/summary.json').read_text())['results'][0]
                with open(Path(summ['run_dir'])/'predictions.csv') as f:
                    pred = {x['pair_id']: int(x['pred']) for x in csv.DictReader(f)}
                D.add(('ditto', s, seed), b, np.array([pred[i] for i in ids[b]], dtype=np.int8))
    LAB = ['RULES', 'FS10', 'RULESFS10_fixed']
    for m in ('pplx_v12', 'jev'):
        for name in LAB:
            prefix, runs = cs.SERIES[name]
            for s in SIZES:
                for r, ns in runs.items():
                    for b in B:
                        D.add(('lab', name, m, s, r), b, journal(ROOT/ns/m/f'{prefix}_{s}'/b/'test.jsonl', ids[b]))
    D.compute()
    draws = RunDraws()

    def cell(a, z, bs):
        e = compare(D, draws, {b: (a, z) for b in bs}, None)
        return tested(e['diff'], e['fixed_ci95'], e['multi_ci95'], e['pairing'])

    def b0(m):
        return ('B0', m), [('B0', m, r) for r in RUNS]

    tables = {}
    # Table V
    rows = {}
    for b in B:
        rows[b] = {}
        for m in P_MODELS:
            p = pr[m]['selection'][b]
            if p == 'B0':
                rows[b][m] = not_tested('DP was selected on validation; both sides are the same answers')
            else:
                rows[b][m] = {**cell((('prompt', m, p), [('prompt', m, p, r) for r in RUNS]), b0(m), [b]), 'prompt': p}
    tables['tab:prompt-selection'] = {'rows': rows, 'Delta Best on test - DP': not_tested('selects on the test set; an upper bound')}
    # Table VI
    rows = {}
    for group, vs in GROUPS.items():
        rows[f'Mean over variants ({group})'] = {}
        for m in inputs_b0.RUN_MODELS:
            def side(v, b, multi):
                return side_reps(D, draws, ('input', m, v), [('input', m, v, r) for r in RUNS], b, multi)-side_reps(D, draws, ('B0', m), [('B0', m, r) for r in RUNS], b, multi)
            res = {}
            for multi in (False, True):
                res[multi] = ci(sum(sum(side(v, b, multi) for b in inputs_b0.benchmarks(v))/len(inputs_b0.benchmarks(v)) for v in vs)/len(vs))
            point = statistics.mean(100*statistics.mean(statistics.mean(D.point[(('input', m, v, r), b)]-D.point[(('B0', m, r), b)] for r in RUNS)
                                                        for b in inputs_b0.benchmarks(v)) for v in vs)
            assert abs(point-statistics.mean(inp[m][v]['diff_mean'] for v in vs)) < 1e-9, (m, group)
            rows[f'Mean over variants ({group})'][m] = tested(point, res[False], res[True], 'independent')
    tables['tab:inputs'] = {'rows': rows, 'combination': 'Per replicate, mean over the variants of the group of the variant minus DP difference, each variant averaged over its own benchmarks (Lowercase: its four benchmarks, one variant among five).'}
    # Table VII
    rows, mean_check = {}, {}
    for m in cs.DM4:
        for c in CONFIGS:
            a, z = (('dm', m), [('casc', m, c, r) for r in RUNS]), (('dm', m), [('alone', m, r) for r in RUNS])
            row = {b: cell(a, z, [b]) for b in B}
            row['mean'] = cell(a, z, B)
            rows[f'{m} + {c}'] = row
            old = fixed_t[(m, c)]['vs_alone']
            assert abs(row['mean']['diff']-old['diff']) < 1e-9, (m, c)
            mean_check[f'{m} + {c}'] = {'stored_significant': old['fixed_significant'] and old['multi_significant'], 'significant_here': row['mean']['significant'],
                                        'stored_fixed_ci95': old['fixed_ci95'], 'stored_multi_ci95': old['multi_ci95']}
    tables['tab:cascade'] = {'rows': rows, 'mean_level_delta_alone_check': mean_check, 't': T}
    # Table IX
    rows = {'PPLX v1.2 (DP)': not_tested('reference row'), 'Jev 1.13 (DP)': not_tested('reference row')}
    pplx0 = (('b0ref', 'pplx_v12'), [('B0', 'pplx_v12', r) for r in RUNS])

    def labeled_row(name, a, z, key):
        row = {b: cell(a, z, [b]) for b in B}
        row['mean'] = cell(a, z, B)
        if key is not None:
            old = lt[key]
            assert abs(row['mean']['diff']-old['diff']) < 1e-9 and max(abs(u-w) for u, w in zip(row['mean']['fixed_ci95'], old['fixed_ci95'])) < 1e-9, key
        rows[name] = row
    labeled_row('GPT-6.1 Sol', (('sol',), [('sol', r) for r in RUNS]), pplx0, ('sol61', 'Sol'))
    for s in SIZES:
        labeled_row(f'Ditto {s}', (('ditto', s), [('ditto', s, x) for x in SEEDS]), pplx0, ('matcher', f'Ditto {s}'))
    NAME = {'pplx_v12': 'PPLX v1.2', 'jev': 'Jev 1.13'}
    SUF = {'RULES': 'Astra rules', 'FS10': '10 examples', 'RULESFS10_fixed': 'rules + 10 examples'}
    for m in ('pplx_v12', 'jev'):
        ref = (('b0ref', m), [('B0', m, r) for r in RUNS])
        for name in LAB:
            for s in SIZES:
                labeled_row(f'{NAME[m]} + {SUF[name]} {s}', (('lab', name, m, s), [('lab', name, m, s, r) for r in RUNS]), ref, (m, f'{name} {s}'))
    z = (('dm', 'pplx_v12'), [('B0', 'pplx_v12', r) for r in RUNS])
    for c, nm in (('sol61', 'Sol'), ('majority', 'ensemble')):
        labeled_row(f'PPLX cascade + {nm}, t=0.05', (('dm', 'pplx_v12'), [('casc', 'pplx_v12', c, r) for r in RUNS]), z, None)
        labeled_row(f'PPLX cascade + {nm}, tuned t', (('dm', 'pplx_v12'), [('tuned', c, r) for r in RUNS]), z, None)
    rows['Ditto all (benchmark training set)'] = not_tested('no per-pair predictions in the project: the row comes from per-seed or mean F1 of the labelling paper (benchmark_report.json, results csv)')
    tables['tab:labeled'] = {'rows': rows, 'reference': 'decision-model rows vs DP of that model; Sol, Ditto and the PPLX cascades vs PPLX v1.2 DP'}

    summary = {}
    for t, x in tables.items():
        cells = []
        for r, row in x['rows'].items():
            if row.get('status') == 'not_tested':
                continue
            for k, v in row.items():
                if isinstance(v, dict) and 'status' in v:
                    cells.append((r, k, v))
        run = [c for c in cells if c[2]['status'] == 'tested']
        per = [c for c in run if c[1] != 'mean']
        summary[t] = {'tested_excluding_mean_column': len(per), 'significant_excluding_mean_column': sum(c[2]['significant'] for c in per),
                      'mean_column_tests': len(run)-len(per), 'mean_column_significant': sum(c[2]['significant'] for c in run if c[1] == 'mean'),
                      'not_tested_cells': sum(c[2]['status'] == 'not_tested' for c in cells),
                      'bootstraps_disagree': [{'row': c[0], 'column': c[1], 'diff': c[2]['diff'], 'fixed_ci95': c[2]['fixed_ci95'], 'multi_ci95': c[2]['multi_ci95']} for c in run if c[2]['bootstraps_disagree']]}
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': 10_000, 'seed': 0, 'run_draw_seed': 1,
           'units': 'differences and intervals in F1 points',
           'inputs_sha256': {n: hashlib.sha256((RES/n).read_bytes()).hexdigest() for n in INPUT_FILES}, 'tables': tables, 'summary': summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1, default=lambda o: o.item())
    print(json.dumps(summary, indent=1), flush=True)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
