"""Multi-bootstrap (Sellam et al., MultiBERTs) as a sensitivity check of every daggered comparison. Read-only, no inference.

Paired bootstrap of the paper (Section 2.4): 10,000 replicates; per benchmark the test pairs are drawn with
replacement (seed 0, protocol order, the resample of inputs_b0_runs123_eval.py and labeled_rules_reps_eval.py); the same
drawn pairs serve both configurations and all runs; the F1 difference is averaged over the runs and the benchmarks.
Multi-bootstrap: in every replicate the three runs of each configuration are also drawn with replacement (rng seed 1,
one draw per replicate shared by all benchmarks, since a run covers all benchmarks). No grouping by records.
Pairing of the run draws:
- independent when the runs of the two sides are separate API calls: prompts vs B0, input variants vs B0, prompt
  selection vs B0 (on a benchmark where the selection is B0 both sides are the same journals and share the draw),
  Astra rules / Ditto seeds vs B0 Run 1;
- joint (the same run index on both sides) when both sides come from the same probabilities of one run: thresholds vs
  t = 0.5;
- a side with a single run stays fixed (B0 Run 1 in the labeled table, logistic regression, few-shot and rules +
  examples with Run 1 only, the cascades with B0 Run 1).
Per comparison: observed difference (F1 points, mean over runs and benchmarks), the stored 95 % interval the paper uses,
the interval with the same pair resample here and fixed runs (fixed_ci95; equals the stored one where the resample is
the same), the multi-bootstrap interval (2.5th and 97.5th percentile) and whether the significance decision (interval
excludes zero) stays. Cascades (B0 Run 1 against Sol / the ensemble, both one run): multi = stored by construction.
Tables: thresholds (6 models x per benchmark / per model), prompts (best zero-shot prompt vs B0), prompt-selection
(validation selection vs B0), inputs (7 models x 9 variants), labeled (rules, 10 examples, rules + 10 examples, LR and
Ditto), cascade. With --table13 also the three-run series of Table 13 (labeled_table13_runs123_20261104.json).

    .venv/bin/python src/multibootstrap_eval.py --paper-sources <paper sources/> --out artifacts/results/multibootstrap_20261104.json
"""
import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import inputs_b0
import protocol
from inputs_b0_eval import B0_RUN1, journal
from inputs_b0_runs123_eval import b0_path, variant_path
from prompt12_runs123_eval import path as prompt_path
from thresholds_runs123_eval import test_path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/'artifacts/results'
LC = ROOT/'runs/labeled_chapter_20261028'
B = list(protocol.BENCHMARKS)
REPS, RUNS = 10_000, (1, 2, 3)
T_MODELS = ['jev', 'clef', 'clef_flash', 'luna_decisions', 'pplx_v12', 'kev']
P_MODELS = ['jev', 'clef', 'clef_flash', 'luna_decisions', 'pplx_v12', 'kev', 'luna']
SIZES = ['200', '2000']
SEEDS = [42, 52, 62]
SERIES13 = {'FS10': ('FS10', {r: f'runs/labeledch_fewshot2_run0{r}_20261031' for r in RUNS}),
            'RULESFS10_fixed': ('RULESFS10', {r: f'runs/labeledch_rulesfs_run0{r}_20261031' for r in RUNS}),
            'RULESFS10_varying': ('RULESFS10', {1: 'runs/labeledch_rulesfs_run01_20261031', 2: 'runs/labeledch_rulesfsrep_run02_20261104',
                                                3: 'runs/labeledch_rulesfsrep_run03_20261104'})}


def ci(x):
    return [100*float(np.percentile(x, 2.5)), 100*float(np.percentile(x, 97.5))]


def sig(c):
    return c is not None and (c[0] > 0 or c[1] < 0)


class Data:
    """Prediction vectors per key and benchmark; replicate F1 for all keys at once."""
    def __init__(self):
        self.ids, self.y = {}, {}
        for b in B:
            cases = json.loads((ROOT/'data'/b/'test.json').read_text())
            self.ids[b] = [c['pair_id'] for c in cases]
            self.y[b] = np.array([c['label'] for c in cases], dtype=np.int8)
        self.pred = {}  # key -> {b: int8 array}

    def add(self, key, b, pred):
        assert pred.shape == self.y[b].shape and set(np.unique(pred)) <= {0, 1}, (key, b)
        self.pred.setdefault(key, {})[b] = pred.astype(np.int8)

    def compute(self):
        rng = np.random.default_rng(0)
        self.point, self.rep = {}, {}
        for b in B:
            n = len(self.y[b])
            idx = rng.integers(0, n, size=(REPS, n))
            w = np.zeros((REPS, n), dtype=np.float32)
            for i in range(REPS):
                w[i] = np.bincount(idx[i], minlength=n)
            del idx
            keys = [k for k in self.pred if b in self.pred[k]]
            y = self.y[b].astype(np.float32)
            cols = []
            for k in keys:
                p = self.pred[k][b].astype(np.float32)
                cols += [y*p, (1-y)*p, y*(1-p)]
            full = np.stack(cols, axis=1)
            c = (w @ full).astype(np.float64)  # counts are exact in float32
            tot = full.sum(axis=0).astype(np.float64)
            for i, k in enumerate(keys):
                tp, fp, fn = c[:, 3*i], c[:, 3*i+1], c[:, 3*i+2]
                self.rep[(k, b)] = 2*tp/np.maximum(2*tp+fp+fn, 1)
                tp, fp, fn = tot[3*i], tot[3*i+1], tot[3*i+2]
                self.point[(k, b)] = 2*tp/max(2*tp+fp+fn, 1)
            del w, c
            print('replicates', b, len(keys), flush=True)


class RunDraws:
    """Per group (one set of runs from the same API calls) an index matrix REPS x 3 of runs drawn with replacement."""
    def __init__(self):
        self.rng, self.draws = np.random.default_rng(1), {}

    def get(self, group, n_runs):
        if (group, n_runs) not in self.draws:
            self.draws[(group, n_runs)] = self.rng.integers(0, n_runs, size=(REPS, 3)) if n_runs > 1 else None
        return self.draws[(group, n_runs)]


def side_reps(D, draws, group, keys, b, multi):
    """Replicate F1 of one side on benchmark b: mean over its runs (fixed) or over three runs drawn with replacement."""
    arr = np.stack([D.rep[(k, b)] for k in keys])  # runs x REPS
    if not multi or len(keys) == 1:
        return arr.mean(axis=0)
    d = draws.get(group, len(keys))
    return np.take_along_axis(arr, d.T, axis=0).mean(axis=0)


def compare(D, draws, spec, stored):
    """spec: {b: ((group A, keys A), (group B, keys B))}; same group on both sides = joint draw."""
    bs = list(spec)
    point = statistics.mean(statistics.mean(D.point[(k, b)] for k in a[1])-statistics.mean(D.point[(k, b)] for k in z[1])
                            for b, (a, z) in spec.items())
    res = {}
    for multi in (False, True):
        d = sum(side_reps(D, draws, a[0], a[1], b, multi)-side_reps(D, draws, z[0], z[1], b, multi) for b, (a, z) in spec.items())/len(bs)
        res['multi' if multi else 'fixed'] = ci(d)
    a0, z0 = next(iter(spec.values()))
    joint = all(a[0] == z[0] for a, z in spec.values())
    out = {'diff': 100*point, 'stored_ci95': stored, 'fixed_ci95': res['fixed'], 'multi_ci95': res['multi'],
           'stored_significant': sig(stored), 'multi_significant': sig(res['multi']),
           'same_decision': sig(stored) == sig(res['multi']), 'fixed_same_as_stored': sig(stored) == sig(res['fixed']),
           'runs': [len(a0[1]), len(z0[1])], 'pairing': 'fixed' if max(len(a0[1]), len(z0[1])) == 1 else
           ('joint' if joint else ('mixed' if any(a[0] == z[0] for a, z in spec.values()) else 'independent')),
           'benchmarks': len(bs)}
    if stored is not None:
        out['fixed_minus_stored_max_abs'] = max(abs(x-y) for x, y in zip(res['fixed'], stored))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper-sources', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--table13', action='store_true', help='add the three-run Table 13 series')
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    D = Data()
    specs = []  # (table, label, spec builder, stored ci)

    # thresholds: tuned t vs 0.5 on the same probabilities of each run (joint draw)
    th = json.loads((RES/'thresholds_runs123_20261103.json').read_text())
    for m in T_MODELS:
        for r in RUNS:
            for b in B:
                rows = {x['pair_id']: x['probability_match'] for x in map(json.loads, test_path(m, b, r).read_text().splitlines())}
                p = np.array([rows[i] for i in D.ids[b]])
                D.add(('t05', m, r), b, p >= .5)
                D.add(('tbench', m, r), b, p >= th['models'][m]['t_benchmark'][b])
                D.add(('tglobal', m, r), b, p >= th['models'][m]['t_global'])
        for kind, key in (('bench', 'tbench'), ('global', 'tglobal')):
            spec = {b: ((('B0prob', m), [(key, m, r) for r in RUNS]), (('B0prob', m), [('t05', m, r) for r in RUNS])) for b in B}
            specs.append(('thresholds', {'model': m, 'threshold': kind}, spec, th['bootstrap'][m][kind]['run_mean']['ci95']))

    # B0 journals (prompts, selection, inputs) and the prompts / variants
    for m in P_MODELS:
        for r in RUNS:
            for b in B:
                D.add(('B0', m, r), b, journal(b0_path(m, b, r), D.ids[b]))
    pr = json.loads((RES/'prompt12_runs123_20261103.json').read_text())
    for m in P_MODELS:
        need = {pr['table9'][m]['selection'][b] for b in B} | {pr['bootstrap']['best_zero_shot_vs_b0'][m]['run_mean']['best']}
        for p in need - {'B0'}:
            for r in RUNS:
                for b in B:
                    D.add(('prompt', m, p, r), b, journal(prompt_path(m, p, b, r), D.ids[b]))
        best = pr['bootstrap']['best_zero_shot_vs_b0'][m]['run_mean']
        if best['best'] != 'B0':
            p = best['best']
            spec = {b: ((('prompt', m, p), [('prompt', m, p, r) for r in RUNS]), (('B0', m), [('B0', m, r) for r in RUNS])) for b in B}
            specs.append(('prompts', {'model': m, 'prompt': p}, spec, best['ci95']))
        sel = pr['table9'][m]['selection']
        spec = {}
        for b in B:
            p = sel[b]
            side = (('B0', m), [('B0', m, r) for r in RUNS]) if p == 'B0' else (('prompt', m, p), [('prompt', m, p, r) for r in RUNS])
            spec[b] = (side, (('B0', m), [('B0', m, r) for r in RUNS]))
        specs.append(('prompt-selection', {'model': m, 'b0_selected': sum(sel[b] == 'B0' for b in B)}, spec,
                      pr['bootstrap']['valid_selection_vs_b0'][m]['run_mean']['ci95']))
    inp = json.loads((RES/'inputs_b0_runs123_20261104.json').read_text())
    inputs_b0.register_models()
    reused = inputs_b0.reused(ROOT)
    for m in inputs_b0.RUN_MODELS:
        for v in inputs_b0.VARIANT_IDS:
            bs = inputs_b0.benchmarks(v)
            for r in RUNS:
                for b in bs:
                    D.add(('input', m, v, r), b, journal(variant_path(reused, m, v, b, r), D.ids[b]))
            spec = {b: ((('input', m, v), [('input', m, v, r) for r in RUNS]), (('B0', m), [('B0', m, r) for r in RUNS])) for b in bs}
            specs.append(('inputs', {'model': m, 'variant': v}, spec, inp['bootstrap'][m][v]['run_mean']['ci95']))

    # labeled: against B0 Run 1 of the same decision model (fixed side), matchers against PPLX v1.2 B0 Run 1
    v2 = json.loads((args.paper_sources/'labeled_v2.json').read_text())
    facts = json.loads((args.paper_sources/'labeled_facts.json').read_text())['vs_pplx_b0']
    for m in ('jev', 'pplx_v12'):
        for b in B:
            D.add(('B0run1', m), b, journal(ROOT/B0_RUN1.get(m, 'runs/quality')/m/'B0'/b/'test.jsonl', D.ids[b]))
        ref = (('B0run1', m), [('B0run1', m)])
        for s in SIZES:
            for r in RUNS:
                for b in B:
                    D.add(('rules', m, s, r), b, journal(ROOT/f'runs/labeledch_rules_run0{r}_20261028'/m/f'RULES_{s}'/b/'test.jsonl', D.ids[b]))
            spec = {b: ((('rules', m, s), [('rules', m, s, r) for r in RUNS]), ref) for b in B}
            specs.append(('labeled', {'model': m, 'method': 'Astra rules', 'size': s}, spec, v2['rules'][m][s]['mean']['ci95']))
            for key, prefix, ns in (('fewshot2', 'FS10', 'runs/labeledch_fewshot2_run01_20261031'),
                                    ('rulesfs', 'RULESFS10', 'runs/labeledch_rulesfs_run01_20261031')):
                v = f'{prefix}_{s}'
                for b in B:
                    D.add((key, m, s), b, journal(ROOT/ns/m/v/b/'test.jsonl', D.ids[b]))
                spec = {b: (((key, m, s), [(key, m, s)]), ref) for b in B}
                specs.append(('labeled', {'model': m, 'method': v + ' (Run 1)', 'size': s}, spec,
                              [100*x for x in v2[key][m][v]['mean']['ci95']]))
    lr = json.loads((LC/'lr/test_predictions.json').read_text())
    ref = (('B0run1', 'pplx_v12'), [('B0run1', 'pplx_v12')])
    for s in SIZES:
        for b in B:
            D.add(('lr', s), b, np.array([lr[b][s]['predictions'][i] for i in D.ids[b]], dtype=np.int8))
            for seed in SEEDS:
                summ = json.loads((LC/f'ditto/output/{b}_set{s}_seed{seed}/summary.json').read_text())['results'][0]
                with open(Path(summ['run_dir'])/'predictions.csv') as f:
                    pred = {x['pair_id']: int(x['pred']) for x in csv.DictReader(f)}
                assert pred.keys() == set(D.ids[b]), (b, s, seed)
                D.add(('ditto', s, seed), b, np.array([pred[i] for i in D.ids[b]], dtype=np.int8))
        specs.append(('labeled', {'model': 'matcher', 'method': 'Logistic regression', 'size': s},
                      {b: ((('lr', s), [('lr', s)]), ref) for b in B}, facts['lr'][s]['ci95']))
        specs.append(('labeled', {'model': 'matcher', 'method': 'Ditto (3 seeds)', 'size': s},
                      {b: ((('ditto', s), [('ditto', s, x) for x in SEEDS]), ref) for b in B}, facts['ditto'][s]['ci95']))
    if args.table13:
        t13 = json.loads((RES/'labeled_table13_runs123_20261104.json').read_text())['summary']
        for name, (prefix, runs) in SERIES13.items():
            for m in ('jev', 'pplx_v12'):
                for s in SIZES:
                    v = f'{prefix}_{s}'
                    for r, ns in runs.items():
                        for b in B:
                            D.add(('t13', name, m, s, r), b, journal(ROOT/ns/m/v/b/'test.jsonl', D.ids[b]))
                    spec = {b: ((('t13', name, m, s), [('t13', name, m, s, r) for r in RUNS]), (('B0run1', m), [('B0run1', m)])) for b in B}
                    specs.append(('labeled-table13', {'model': m, 'method': name, 'size': s}, spec, t13[name][m][s]['ci95']))

    D.compute()
    draws = RunDraws()
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': REPS,
           'units': 'diff and intervals in F1 points', 'comparisons': []}
    for table, label, spec, stored in specs:
        out['comparisons'].append({'table': table, **label, **compare(D, draws, spec, stored)})
    cas = json.loads((args.paper_sources/'cascade_b0.json').read_text())['bootstrap']['cascade_vs_alone']['A_b0']
    for m in ('jev', 'pplx_v12', 'clef', 'luna_decisions'):
        for c in ('sol61', 'majority'):
            x = cas[m]['0.005'][c]
            out['comparisons'].append({'table': 'cascade', 'model': m, 'partner': c, 'diff': x['diff'], 'stored_ci95': x['ci95'],
                                       'fixed_ci95': x['ci95'], 'multi_ci95': x['ci95'], 'stored_significant': sig(x['ci95']),
                                       'multi_significant': sig(x['ci95']), 'same_decision': True, 'fixed_same_as_stored': True,
                                       'runs': [1, 1], 'pairing': 'fixed (one run per side: multi = stored)', 'benchmarks': len(B)})
    summ = {}
    for c in out['comparisons']:
        t = summ.setdefault(c['table'], {'comparisons': 0, 'daggers_stored': 0, 'daggers_kept': 0, 'daggers_lost': [], 'daggers_gained': []})
        t['comparisons'] += 1
        t['daggers_stored'] += c['stored_significant']
        t['daggers_kept'] += c['stored_significant'] and c['multi_significant']
        name = ' / '.join(str(c[k]) for k in ('model', 'threshold', 'prompt', 'variant', 'method', 'size', 'partner') if k in c)
        if c['stored_significant'] and not c['multi_significant']:
            t['daggers_lost'].append(name)
        if c['multi_significant'] and not c['stored_significant']:
            t['daggers_gained'].append(name)
    out['summary'] = {'per_table': summ,
                      'daggers_stored': sum(t['daggers_stored'] for t in summ.values()),
                      'daggers_kept': sum(t['daggers_kept'] for t in summ.values()),
                      'daggers_lost': sum(len(t['daggers_lost']) for t in summ.values()),
                      'daggers_gained': sum(len(t['daggers_gained']) for t in summ.values()),
                      'fixed_differs_from_stored_decision': [c for c in out['comparisons'] if not c['fixed_same_as_stored']]}
    print(json.dumps(out['summary'], indent=1), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
