"""Multi-bootstrap check of every significance statement of the paper, working copy of 10 Nov 2026. Read-only, no inference.

Method of multibootstrap_eval.py (4 Nov): paired bootstrap, 10,000 replicates, seed 0, per benchmark the test pairs drawn with replacement, the same pairs for
both configurations and all runs, difference averaged over the runs and benchmarks (fixed_ci95); multi_ci95 also draws the three runs of each configuration
with replacement (rng seed 1); a side with one run stays fixed. Run draws are joint when both sides use the same probabilities of run r (group ('dm', m)) and
independent when they are separate API calls.
stored_significant: the mark the paper prints or the statement its text makes. stored_ci95: the interval behind it. multi_significant: multi_ci95 excludes zero
(for the statement that only one prompt exceeds DP: and the difference is positive).
Tables: baseline (Delta to the frontier mean), prompt-selection (cells and mean row), inputs (63 cells, 14 mean rows), cascade (cells, Delta alone, Delta random,
Delta similar), labeled (cells and means), thresholds (Sol 2,000). Text: zero-shot prompts vs DP, cascade vs Sol alone and vs ensemble alone, Sol cascade vs
ensemble cascade, tuned vs fixed t, target risks 0.25 % and 1 % vs 0.5 %, rules + examples in the cascade.

    .venv/bin/python src/multibootstrap_paper_20261110_eval.py --paper-sources <dir with baseline.json> --out artifacts/results/multibootstrap_20261110.json
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
from cascade_random_deferral_eval import RUN_SEED
from inputs_b0_eval import journal
from inputs_b0_runs123_eval import b0_path, variant_path
from multibootstrap_eval import LC, P_MODELS, REPS, SEEDS, SIZES, Data, RunDraws, ci, compare, side_reps, sig
from prompt12_runs123_eval import ZERO_SHOT, path as prompt_path
from sota_runs123_eval import later_run
from thresholds_runs123_eval import MODELS as DM6

B, RUNS, RES, ROOT = cs.B, cs.RUNS, cs.RES, cs.ROOT
T = .05
CONFIGS = ('sol61', 'majority')
SOTA = ['sol61', 'opus55', 'argon4']
TABLE2 = ['jev', 'pplx_v12', 'clef', 'clef_flash', 'kev', 'luna_decisions', 'luna', 'qwen38']
GROUPS = {'Serialization': ['F1_values', 'F2_json', 'F3_aligned', 'F4_no_missing'],
          'Perturbation': ['R1_swap', 'R2_attr_reverse', 'R3_lowercase', 'R4_attr_del', 'R6_typos']}
FILES = ['prompt12_runs123_20261103.json', 'inputs_b0_runs123_20261104.json', 'cascade_fixed_t_20261107.json', 'cascade_sol2000_runs123_20261105_v2.json',
         'later_tables_sol2000_runs123_20261105_v2.json', 'sota_runs123_20261104.json', 'table_cell_significance_20261110.json', 'thresholds_sol2000_20261105.json',
         'thresholds_sol2000_multi_frontier_20261106.json', 'cascade_random_deferral_t005_20261107.json', 'cascade_similarity_deferral_20261108.json',
         'similarity_scores_20261108.json', 'cascade_sol2000_tests_20261105.json', 'cascade_sol_vs_ensemble_20261109.json', 'cascade_fixed_t_supplement_20261107.json']


def load(name):
    return json.loads((RES/name).read_text())


def path(m, b, r):
    return ROOT/f'runs/qwen38_b0_run0{r}_20261022/qwen38/B0'/b/'test.jsonl' if m == 'qwen38' else b0_path(m, b, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper-sources', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    pr = load(FILES[0])
    inp = load(FILES[1])
    fixed = {(x['model'], x['partner'], x['t']): x for x in load(FILES[2])['cells']}
    tuned = load(FILES[3])
    lt = {(s['model'], s['row']): s for s in load(FILES[4])['significance'] if s['table'] == 'table10'}
    sota = load(FILES[5])['f1']
    cells = load(FILES[6])['tables']
    th = load(FILES[7])
    fr = {x['model']: x for x in load(FILES[8])['frontier'] if x['benchmarks'] == 'all eight'}
    rnd = {(x['model'], x['partner']): x for x in load(FILES[9])['cells']}
    sim_file = load(FILES[10])
    simc = {(x['model'], x['partner']): x for x in sim_file['cells'] if x['variant'] == 'high_0.16'}
    scores = load(FILES[11])
    t_tests = load(FILES[12])['target_vs_05']
    solens = load(FILES[13])['models']
    supp = load(FILES[14])
    base = json.loads((args.paper_sources/'baseline.json').read_text())['f1']

    D = Data()
    ids, y = D.ids, D.y
    # decision models, generative comparison models, SOTA
    for m in sorted(set(P_MODELS) | set(TABLE2)):
        for r in RUNS:
            for b in B:
                D.add(('B0', m, r), b, journal(path(m, b, r), ids[b]))
    dec1 = cv.ensemble()
    for s in SOTA:
        for r in RUNS:
            pred = {b: np.array([dec1[s][(b, i)][0] for i in ids[b]], dtype=np.int8) for b in B} if r == 1 else later_run(s, r, ids)[0]
            for b in B:
                D.add(('sota', s, r), b, pred[b])
    # prompts
    for m in P_MODELS:
        assert {pr['table9'][m]['selection'][b] for b in B} <= set(ZERO_SHOT), m
        for p in ZERO_SHOT:
            if p != 'B0':
                for r in RUNS:
                    for b in B:
                        D.add(('prompt', m, p, r), b, journal(prompt_path(m, p, b, r), ids[b]))
    # input variants
    inputs_b0.register_models()
    reused = inputs_b0.reused(ROOT)
    for m in inputs_b0.RUN_MODELS:
        for v in inputs_b0.VARIANT_IDS:
            for r in RUNS:
                for b in inputs_b0.benchmarks(v):
                    D.add(('input', m, v, r), b, journal(variant_path(reused, m, v, b, r), ids[b]))
    # thresholds on the Sol 2,000 set
    for m in DM6:
        for r in RUNS:
            for b in B:
                got = {x['pair_id']: x['probability_match'] for x in cv.lines(cs.b0_path(m, b, r))}
                p = np.array([got[i] for i in ids[b]])
                D.add(('t05', m, r), b, (p >= .5).astype(np.int8))
                D.add(('tbench', m, r), b, (p >= th['models'][m]['t_benchmark'][b]).astype(np.int8))
                D.add(('tglobal', m, r), b, (p >= th['models'][m]['t_global']).astype(np.int8))
    # cascades
    partner = {r: cs.partner_run(r, ids, dec1)[0] for r in RUNS}
    for r in RUNS:
        for b in B:
            D.add(('sol', r), b, partner[r]['sol61'][b][0])
            D.add(('ens', r), b, partner[r]['majority'][b][0])
    sim_low = sim_file['thresholds']['high_0.16']['low']
    sim_mask = {b: np.array([sum(scores[b]['test'][i])/2 >= sim_low for i in ids[b]]) for b in B}
    perm = {(r, b): np.random.default_rng([RUN_SEED[r], i]).permutation(len(ids[b])) for r in RUNS for i, b in enumerate(B)}
    for m in cs.DM4:
        for r in RUNS:
            test = cs.test_rows(lambda b: cs.b0_path(m, b, r), ids)
            _, preds = cs.cascade(y, test, partner[r], T)
            for b in B:
                alone = preds['alone'][b]
                D.add(('alone', m, r), b, alone)
                for c in CONFIGS:
                    D.add(('casc', m, c, r), b, preds[c][b])
                    pc = partner[r][c][b][0]
                    D.add(('sim', m, c, r), b, np.where(sim_mask[b], pc, alone).astype(np.int8))
                    d = np.zeros(len(alone), dtype=bool)
                    d[perm[(r, b)][:int((np.minimum(test[b]['p'], 1-test[b]['p']) > T).sum())]] = True
                    D.add(('rand', m, c, r), b, np.where(d, pc, alone).astype(np.int8))
            for x in cs.TARGETS:
                _, px = cs.cascade(y, test, partner[r], tuned['targets'][m][str(x)]['t'])
                for c in CONFIGS:
                    for b in B:
                        D.add(('tx', m, x, c, r), b, px[c][b])
    # labeled rows
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
    prefix, runs = cs.SERIES['RULESFS10_fixed']
    for r in RUNS:
        test = cs.test_rows(lambda b: ROOT/runs[r]/'pplx_v12/RULESFS10_2000'/b/'test.jsonl', ids)
        _, px = cs.cascade(y, test, partner[r], T)
        for c in CONFIGS:
            for b in B:
                D.add(('labc', c, r), b, px[c][b])
    D.compute()
    for m in TABLE2:
        for r in RUNS:
            assert all(abs(D.point[(('B0', m, r), b)]-base[m][b][str(r)]) < 1e-12 for b in B), (m, r)
    for s in SOTA:
        for r in RUNS:
            assert all(abs(D.point[(('sota', s, r), b)]-sota[s][b][str(r)]) < 1e-12 for b in B), (s, r)
    draws = RunDraws()
    out = []

    def side(group, fmt):
        return group, [fmt(r) for r in RUNS]

    def push(table, label, spec, stored, printed, stored_diff=None, positive_only=False):
        e = compare(D, draws, spec, stored)
        if stored_diff is not None:
            assert abs(e['diff']-stored_diff) < 1e-6, (table, label, e['diff'], stored_diff)
        multi = sig(e['multi_ci95']) and (not positive_only or e['diff'] > 0)
        e.update(table=table, **label, stored_significant=bool(printed), multi_significant=bool(multi), same_decision=bool(printed) == bool(multi))
        e['fixed_significant'] = sig(e['fixed_ci95']) and (not positive_only or e['diff'] > 0)
        e['fixed_same_as_stored'] = bool(printed) == e['fixed_significant']
        out.append(e)

    def spec_of(a, z, bs=B):
        return {b: (a, z) for b in bs}

    def b0(m):
        return side(('B0', m), lambda r: ('B0', m, r))

    # Table II: Delta to the mean of the frontier LLMs
    for m in TABLE2:
        ref = statistics.mean(100*statistics.mean(statistics.mean(D.point[(('sota', s, r), b)] for r in RUNS) for b in B) for s in SOTA)
        mean_m = 100*statistics.mean(statistics.mean(D.point[(('B0', m, r), b)] for r in RUNS) for b in B)

        def reps(kind, x, multi):
            return sum(side_reps(D, draws, (kind, x), [(kind, x, r) for r in RUNS], b, multi) for b in B)/len(B)
        res = {mu: ci(reps('B0', m, mu)-sum(reps('sota', s, mu) for s in SOTA)/len(SOTA)) for mu in (False, True)}
        old = fr[m]
        assert abs(mean_m-ref-old['diff']) < 1e-6, m
        printed = old['fixed_significant'] and old['multi_significant']
        out.append({'table': 'baseline', 'model': m, 'comparison': 'minus mean of Sol, Opus, Argon', 'diff': mean_m-ref, 'stored_ci95': old['fixed_ci95'], 'fixed_ci95': res[False],
                    'multi_ci95': res[True], 'stored_significant': printed, 'multi_significant': sig(res[True]), 'same_decision': printed == sig(res[True]),
                    'fixed_significant': sig(res[False]), 'fixed_same_as_stored': printed == sig(res[False]), 'runs': [3, 9], 'pairing': 'independent', 'benchmarks': len(B)})
    # Table V
    for m in P_MODELS:
        sel = pr['table9'][m]['selection']
        for b in B:
            if sel[b] == 'B0':
                continue
            p, c = sel[b], cells['tab:prompt-selection']['rows'][b][m]
            push('prompt-selection', {'model': m, 'benchmark': b, 'prompt': p}, spec_of(side(('prompt', m, p), lambda r: ('prompt', m, p, r)), b0(m), [b]), c['fixed_ci95'], c['significant'], c['diff'])
        spec = {b: (b0(m) if sel[b] == 'B0' else side(('prompt', m, sel[b]), lambda r: ('prompt', m, sel[b], r)), b0(m)) for b in B}
        s = pr['bootstrap']['valid_selection_vs_b0'][m]['run_mean']
        push('prompt-selection', {'model': m, 'benchmark': 'mean'}, spec, s['ci95'], sig(s['ci95']))
    # Table VI
    for m in inputs_b0.RUN_MODELS:
        for v in inputs_b0.VARIANT_IDS:
            s = inp['bootstrap'][m][v]['run_mean']
            push('inputs', {'model': m, 'variant': v}, spec_of(side(('input', m, v), lambda r: ('input', m, v, r)), b0(m), inputs_b0.benchmarks(v)), s['ci95'], sig(s['ci95']), inp['summary'][m][v]['diff_mean'])
        for g, vs in GROUPS.items():
            def rep(multi):
                return sum(sum(side_reps(D, draws, ('input', m, v), [('input', m, v, r) for r in RUNS], b, multi)-side_reps(D, draws, ('B0', m), [('B0', m, r) for r in RUNS], b, multi)
                               for b in inputs_b0.benchmarks(v))/len(inputs_b0.benchmarks(v)) for v in vs)/len(vs)
            old = cells['tab:inputs']['rows'][f'Mean over variants ({g})'][m]
            fx, mu = ci(rep(False)), ci(rep(True))
            out.append({'table': 'inputs', 'model': m, 'variant': f'Mean over variants ({g})', 'diff': old['diff'], 'stored_ci95': old['fixed_ci95'], 'fixed_ci95': fx, 'multi_ci95': mu,
                        'stored_significant': old['significant'], 'multi_significant': sig(mu), 'same_decision': old['significant'] == sig(mu), 'fixed_significant': sig(fx),
                        'fixed_same_as_stored': old['significant'] == sig(fx), 'runs': [3, 3], 'pairing': 'independent', 'benchmarks': len(B)})
    # Table VII
    for m in cs.DM4:
        a_alone = side(('dm', m), lambda r: ('alone', m, r))
        for c in CONFIGS:
            real = side(('dm', m), lambda r: ('casc', m, c, r))
            row = cells['tab:cascade']['rows'][f'{m} + {c}']
            for b in B:
                push('cascade', {'model': m, 'partner': c, 'benchmark': b, 'comparison': 'vs alone'}, spec_of(real, a_alone, [b]), row[b]['fixed_ci95'], row[b]['significant'], row[b]['diff'])
            x = fixed[(m, c, T)]
            push('cascade', {'model': m, 'partner': c, 'benchmark': 'mean', 'comparison': 'Delta alone'}, spec_of(real, a_alone), x['vs_alone']['fixed_ci95'], sig(x['vs_alone']['fixed_ci95']), x['vs_alone']['diff'])
            r0 = rnd[(m, c)]['real_minus_random']
            push('cascade', {'model': m, 'partner': c, 'benchmark': 'mean', 'comparison': 'Delta random'}, spec_of(real, side(('dm', m), lambda r: ('rand', m, c, r))), r0['fixed_ci95'], sig(r0['fixed_ci95']), r0['diff'])
            s0 = simc[(m, c)]['real_minus_similarity']
            push('cascade', {'model': m, 'partner': c, 'benchmark': 'mean', 'comparison': 'Delta similar (high_0.16)'}, spec_of(real, side(('dm', m), lambda r: ('sim', m, c, r))), s0['fixed_ci95'], sig(s0['fixed_ci95']), s0['diff'])
    # Table IX
    rows = cells['tab:labeled']['rows']
    pplx0 = side(('b0ref', 'pplx_v12'), lambda r: ('B0', 'pplx_v12', r))

    def lab_row(name, a, z, key):
        row = rows[name]
        for b in B:
            push('labeled', {'row': name, 'benchmark': b}, spec_of(a, z, [b]), row[b]['fixed_ci95'], row[b]['significant'], row[b]['diff'])
        ci_stored = lt[key]['fixed_ci95'] if key is not None else row['mean']['fixed_ci95']
        printed = sig(ci_stored) if key is not None else row['mean']['significant']
        push('labeled', {'row': name, 'benchmark': 'mean'}, spec_of(a, z), ci_stored, printed, row['mean']['diff'])
    lab_row('GPT-6.1 Sol', side(('sol',), lambda r: ('sol', r)), pplx0, ('sol61', 'Sol'))
    for s in SIZES:
        lab_row(f'Ditto {s}', (('ditto', s), [('ditto', s, x) for x in SEEDS]), pplx0, ('matcher', f'Ditto {s}'))
    NAME, SUF = {'pplx_v12': 'PPLX v1.2', 'jev': 'Jev 1.13'}, {'RULES': 'Astra rules', 'FS10': '10 examples', 'RULESFS10_fixed': 'rules + 10 examples'}
    for m in NAME:
        ref = side(('b0ref', m), lambda r: ('B0', m, r))
        for name in LAB:
            for s in SIZES:
                lab_row(f'{NAME[m]} + {SUF[name]} {s}', (('lab', name, m, s), [('lab', name, m, s, r) for r in RUNS]), ref, (m, f'{name} {s}'))
    z = side(('dm', 'pplx_v12'), lambda r: ('B0', 'pplx_v12', r))
    for c, nm in (('sol61', 'Sol'), ('majority', 'ensemble')):
        lab_row(f'PPLX cascade + {nm}, t=0.05', side(('dm', 'pplx_v12'), lambda r: ('casc', 'pplx_v12', c, r)), z, None)
        lab_row(f'PPLX cascade + {nm}, tuned t', side(('dm', 'pplx_v12'), lambda r: ('tx', 'pplx_v12', cs.X, c, r)), z, None)
    # Table VIII
    for m in DM6:
        for kind, key in (('bench', 'tbench'), ('global', 'tglobal')):
            s = th['bootstrap'][m][kind]['run_mean']
            push('thresholds', {'model': m, 'threshold': kind}, spec_of(side(('dm', m), lambda r: (key, m, r)), side(('dm', m), lambda r: ('t05', m, r))), s['ci95'], sig(s['ci95']), th['models'][m]['over_runs'][f'diff_{kind}_mean'])
    # Text
    best = {m: pr['bootstrap']['best_zero_shot_vs_b0'][m]['run_mean'] for m in P_MODELS if 'run_mean' in pr['bootstrap']['best_zero_shot_vs_b0'].get(m, {})}
    for m in P_MODELS:
        for p in ZERO_SHOT:
            if p == 'B0':
                continue
            is_best = m in best and best[m].get('best') == p
            stored = best[m]['ci95'] if is_best else None
            push('text: prompts vs DP', {'model': m, 'prompt': p}, spec_of(side(('prompt', m, p), lambda r: ('prompt', m, p, r)), b0(m)), stored, m == 'kev' and p == 'PE_general_simple', positive_only=True)
    for m in cs.DM4:
        real = {c: side(('dm', m), lambda r: ('casc', m, c, r)) for c in CONFIGS}
        for c, grp in (('sol61', 'sol'), ('majority', 'ens')):
            x = fixed[(m, c, T)]['vs_partner_alone']
            push('text: cascade vs partner alone', {'model': m, 'partner': c}, spec_of(real[c], side(('dm', m), lambda r: (grp, r))), x['fixed_ci95'], sig(x['fixed_ci95']), x['diff'])
        v = solens[m]['sol_cascade_minus_ensemble_cascade']
        push('text: Sol cascade vs ensemble cascade', {'model': m}, spec_of(real['sol61'], real['majority']), v['fixed_ci95'], sig(v['fixed_ci95']), v['diff'])
        for c in CONFIGS:
            x = fixed[(m, c, T)]['vs_tuned']
            s = {'diff': -x['diff'], 'ci': [-x['fixed_ci95'][1], -x['fixed_ci95'][0]]}
            push('text: tuned vs t=0.05', {'model': m, 'partner': c}, spec_of(side(('dm', m), lambda r: ('tx', m, cs.X, c, r)), real[c]), s['ci'], sig(s['ci']), s['diff'])
    for e in t_tests:
        m, x, c = e['model'], e['target_risk'], e['config']
        push('text: target risk vs 0.5 %', {'model': m, 'partner': c, 'target_risk': x}, spec_of(side(('dm', m), lambda r: ('tx', m, x, c, r)), side(('dm', m), lambda r: ('tx', m, cs.X, c, r))), e['fixed_ci95'], sig(e['fixed_ci95']), e['diff'])
    lab_group = ('lab', 'RULESFS10_fixed', 'pplx_v12', '2000')
    for s in supp['literature']['significance']:
        if s['comparison'].startswith('RULESFS10_2000 alone'):
            continue
        c = 'sol61' if ' sol61 ' in s['comparison'] else 'majority'
        push('text: rules + examples in the cascade', {'partner': c}, spec_of(side(lab_group, lambda r: ('labc', c, r)), side(('dm', 'pplx_v12'), lambda r: ('casc', 'pplx_v12', c, r))), s['fixed_ci95'], sig(s['fixed_ci95']), s['diff'])

    table_keys = sorted({c['table'] for c in out})
    per, lost, gained = {}, [], []
    for t in table_keys:
        rows_t = [c for c in out if c['table'] == t]
        per[t] = {'comparisons': len(rows_t), 'daggers_stored': sum(c['stored_significant'] for c in rows_t), 'daggers_kept': sum(c['stored_significant'] and c['multi_significant'] for c in rows_t),
                  'daggers_lost': sum(c['stored_significant'] and not c['multi_significant'] for c in rows_t), 'daggers_gained': sum(c['multi_significant'] and not c['stored_significant'] for c in rows_t)}
    for c in out:
        if c['stored_significant'] and not c['multi_significant']:
            lost.append(c)
        if c['multi_significant'] and not c['stored_significant']:
            gained.append(c)
    summary = {'per_table': per, 'daggers_stored': sum(p['daggers_stored'] for p in per.values()), 'daggers_kept': sum(p['daggers_kept'] for p in per.values()),
               'daggers_lost': len(lost), 'daggers_gained': len(gained), 'lost': lost, 'gained': gained,
               'fixed_differs_from_stored_decision': [c for c in out if not c['fixed_same_as_stored']]}
    res = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': REPS, 'seed': 0, 'run_draw_seed': 1, 'units': 'diff and intervals in F1 points',
           'inputs_sha256': {n: hashlib.sha256((RES/n).read_bytes()).hexdigest() for n in FILES}, 'comparisons': out, 'summary': summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(res, f, indent=1, default=lambda o: o.item())
    print(json.dumps({k: v for k, v in summary.items() if k not in ('lost', 'gained', 'fixed_differs_from_stored_decision')}, indent=1), flush=True)
    print('lost', [(c['table'], {k: c.get(k) for k in ('model', 'partner', 'benchmark', 'variant', 'row', 'threshold', 'prompt', 'comparison', 'target_risk')}, round(c['diff'], 2), [round(v, 2) for v in c['fixed_ci95']], [round(v, 2) for v in c['multi_ci95']]) for c in lost], flush=True)
    print('gained', [(c['table'], {k: c.get(k) for k in ('model', 'partner', 'benchmark', 'variant', 'row', 'threshold', 'prompt', 'comparison', 'target_risk')}, round(c['diff'], 2), [round(v, 2) for v in c['fixed_ci95']], [round(v, 2) for v in c['multi_ci95']]) for c in gained], flush=True)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
