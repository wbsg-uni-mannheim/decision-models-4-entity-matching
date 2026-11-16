"""Scores the labelled-chapter Jev/PPLX runs against B0 Run 1. Read-only, no inference.

--group rules: labeledch_rules_run01_20261028, RULES_200/2000/4000 = B0 + Astra rules from the Sol-B0 labelled set.
--group fewshot: labeledch_fewshot_run01_20261028, FS10/FS20 = B0 + 10/20 retrieved examples from the 4000 set.
--group fewshot2: labeledch_fewshot2_run01_20261031, FS10_200/FS10_2000 = B0 + 10 examples from the 200 / 2000 set,
  duplicate filter (configs/labeled_chapter_fewshot_v2_20261031.json; its per-benchmark filter counts are copied).
--group rulesfs: labeledch_rulesfs_run01_20261031, RULESFS10_200/2000 = B0 + run-1 Astra rules + the FS10_<n> examples.
Per model, benchmark and variant: F1 of the match class, F1 of B0 Run 1, difference and flip rate (share of test
pairs whose decision differs from B0).
Paired bootstrap (10,000 replicates, seed 0, same resample for B0 and the variant; method of inputs_b0_eval.py):
95% CI of the difference per cell and of the difference averaged over the 8 benchmarks per model and variant.

    .venv/bin/python src/labeled_rules_eval.py --out artifacts/results/labeled_rules_20261028.json
    .venv/bin/python src/labeled_rules_eval.py --group fewshot --out artifacts/results/labeled_fewshot_20261028.json
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import protocol
from inputs_b0_eval import B0_RUN1, f1, journal

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {'rules': ('runs/labeledch_rules_run01_20261028', ['RULES_200', 'RULES_2000', 'RULES_4000']),
          'fewshot': ('runs/labeledch_fewshot_run01_20261028', ['FS10', 'FS20']),
          'fewshot2': ('runs/labeledch_fewshot2_run01_20261031', ['FS10_200', 'FS10_2000']),
          'rulesfs': ('runs/labeledch_rulesfs_run01_20261031', ['RULESFS10_200', 'RULESFS10_2000'])}
RETRIEVAL_V2 = 'configs/labeled_chapter_fewshot_v2_20261031.json'
MODELS = ['jev', 'pplx_v12']
REPS = 10_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', choices=list(GROUPS), default='rules')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    namespace, variants = GROUPS[args.group]
    rng = np.random.default_rng(0)
    data = {}
    for b in protocol.BENCHMARKS:
        cases = json.loads((ROOT/'data'/b/'test.json').read_text())
        ids = [c['pair_id'] for c in cases]
        data[b] = (ids, np.array([c['label'] for c in cases], dtype=np.int8), rng.integers(0, len(ids), size=(REPS, len(ids))))
    out = {'description': __doc__, 'group': args.group, 'replicates': REPS, 'cells': {}, 'mean': {},
           'created_at': datetime.now(timezone.utc).isoformat()}
    for m in MODELS:
        diffs = {v: np.zeros(REPS) for v in variants}
        for b in protocol.BENCHMARKS:
            ids, y, sample = data[b]
            full = np.arange(len(ids))
            b0 = journal(ROOT/B0_RUN1.get(m, 'runs/quality')/m/'B0'/b/'test.jsonl', ids)
            f0, f0_s = float(f1(y, b0, full)), f1(y, b0, sample)
            for v in variants:
                var = journal(ROOT/namespace/m/v/b/'test.jsonl', ids)
                fv, d = float(f1(y, var, full)), f1(y, var, sample)-f0_s
                diffs[v] += d/len(protocol.BENCHMARKS)
                out['cells'].setdefault(m, {}).setdefault(b, {})[v] = {
                    'f1': fv, 'f1_b0': f0, 'diff': fv-f0, 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    'flip': float((var != b0).mean()), 'n': len(ids)}
        for v in variants:
            cells = [out['cells'][m][b][v] for b in protocol.BENCHMARKS]
            out['mean'].setdefault(m, {})[v] = {**{k: float(np.mean([c[k] for c in cells])) for k in ('f1', 'f1_b0', 'diff')},
                                                'ci95': [float(np.percentile(diffs[v], 2.5)), float(np.percentile(diffs[v], 97.5))]}
        print(m, {v: round(out['mean'][m][v]['diff'], 4) for v in variants}, flush=True)
    if args.group in ('fewshot2', 'rulesfs'):
        r = json.loads((ROOT/RETRIEVAL_V2).read_text())
        out['duplicate_filter'] = {'retrieval': RETRIEVAL_V2, 'rule': r['filter'],
                                   'per_benchmark': {b: {n: {k: v for k, v in x.items() if k != 'k10'} for n, x in r['benchmarks'][b].items()}
                                                     for b in protocol.BENCHMARKS}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
