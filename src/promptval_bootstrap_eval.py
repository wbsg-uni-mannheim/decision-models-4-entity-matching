"""Paired bootstrap of the validation-selected prompt against B0 on test, per model. Read-only, no inference.

Selection from artifacts/results/prompt_validation_20261027.json (selection[m][b].prompt). Test journals as in
promptval_eval.test_f1: prompt12_run01_20261019, B0 from the baseline table (runs/quality) except the prompt12 extra
models (pplx_v12), which use their own B0. Reference arm: test oracle = best of the ten zero-shot prompts per benchmark
on test (ties: promptval.PROMPTS order, B0 first). Method of inputs_b0_eval.py: stratified per benchmark, 10,000
replicates, seed 0, same resample for B0 and the arm; mean F1 difference over the 8 benchmarks in points, 95% CI,
correct->wrong and wrong->correct counts against B0. Point diffs are checked against prompt_validation_20261027.json.

    .venv/bin/python src/promptval_bootstrap_eval.py --out artifacts/results/prompt_validation_bootstrap_20261030.json
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import prompt12
import promptval
import protocol
from inputs_b0_eval import f1, journal
from promptval_eval import TEST_NAMESPACE

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT/'artifacts/results/prompt_validation_20261027.json'
REPS = 10_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    promptval.register_models()
    pv = json.loads(SELECTION.read_text())
    rng = np.random.default_rng(0)
    data = {}
    for b in protocol.BENCHMARKS:
        cases = json.loads((ROOT/'data'/b/'test.json').read_text())
        ids = [c['pair_id'] for c in cases]
        data[b] = (ids, np.array([c['label'] for c in cases], dtype=np.int8), rng.integers(0, len(ids), size=(REPS, len(ids))))
    out = {'description': __doc__, 'replicates': REPS, 'selection_file': str(SELECTION.relative_to(ROOT)), 'cells': {},
           'bootstrap': {}, 'created_at': datetime.now(timezone.utc).isoformat()}
    for m in promptval.RUN_MODELS:
        def path(v, b):
            own = v != 'B0' or m in prompt12.EXTRA_MODELS
            return ROOT/'runs'/(TEST_NAMESPACE if own else 'quality')/m/v/b/'test.jsonl'
        arms = {a: {'diff': np.zeros(REPS), 'point': 0., 'worse': 0, 'better': 0} for a in ('valid_selected', 'test_oracle')}
        for b in protocol.BENCHMARKS:
            ids, y, sample = data[b]
            full = np.arange(len(ids))
            preds = {v: journal(path(v, b), ids) for v in promptval.PROMPTS}
            test = {v: float(f1(y, p, full)) for v, p in preds.items()}
            sel = pv['selection'][m][b]
            assert abs(test[sel['prompt']]-sel['test_f1']) < 1e-12 and abs(test['B0']-sel['b0_test_f1']) < 1e-12, (m, b)
            prompts = {'valid_selected': sel['prompt'], 'test_oracle': max(promptval.PROMPTS, key=lambda p: (test[p], -promptval.PROMPTS.index(p)))}
            b0, b0_s = preds['B0'], f1(y, preds['B0'], sample)
            for a, v in prompts.items():
                p, s = arms[a], preds[v]
                p['point'] += (test[v]-test['B0'])/len(protocol.BENCHMARKS)
                p['diff'] += (f1(y, s, sample)-b0_s)/len(protocol.BENCHMARKS)
                p['worse'] += int(((b0 == y) & (s != y)).sum()); p['better'] += int(((b0 != y) & (s == y)).sum())
                out['cells'].setdefault(m, {}).setdefault(b, {})[a] = {'prompt': v, 'f1': test[v], 'f1_b0': test['B0'],
                                                                      'diff': 100*(test[v]-test['B0']), 'flip': float((s != b0).mean())}
        means = pv['mean_test_f1_per_model'][m]
        expected = {'valid_selected': means['test_f1']-means['b0_test_f1'], 'test_oracle': means['test_oracle_10_prompts_f1']-means['b0_test_f1']}
        for a, p in arms.items():
            assert abs(p['point']-expected[a]) < 1e-12, (m, a)
            out['bootstrap'].setdefault(m, {})[a] = {'diff': 100*p['point'], 'ci95': [100*float(np.percentile(p['diff'], 2.5)),
                                                                                     100*float(np.percentile(p['diff'], 97.5))],
                                                     'correct_to_wrong': p['worse'], 'wrong_to_correct': p['better'],
                                                     'diff_matches_prompt_validation': True}
        print(m, {a: (round(r['diff'], 2), [round(x, 2) for x in r['ci95']]) for a, r in out['bootstrap'][m].items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
