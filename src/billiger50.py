"""Authorized additive B0 comparison on the Labeling paper's 50%-unseen split.

This module never extends protocol.BENCHMARKS or changes AP2 selection.
Run only after the core job finishes successfully; share its $100 ledger.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

from protocol import MODELS, canonical, digest
from prepare import case, read_rows, write_frozen
from run import PROJECT, run_cell
from orchestrate import stamp
from analyze import load, metric, calibration, bootstrap_delta, csv_save

BENCHMARK = 'billiger-de-50un'
SOURCE_HASH = 'a6973081c59930a1331df0909eeed96b8351ddb54899d9bb4936360c9bcc5204'
PAPER = 'https://arxiv.org/html/2606.28823v2#S4.T7'
HISTORICAL = {'GPT-5.2': 75.71, 'Qwen 3.6 Plus': 69.11, 'Kimi K2.6': 76.12}


def prepare(source):
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_HASH, 'Source checksum mismatch'
    rows = [case(row, 'billiger-de') for row in read_rows(source)]
    assert len(rows) == 4437 and sum(r['label'] for r in rows) == 361
    assert len({r['pair_id'] for r in rows}) == len(rows)
    for row in rows:
        row['benchmark'] = BENCHMARK
    write_frozen(PROJECT/'data'/BENCHMARK/'test.json', rows)
    dest = PROJECT/'data/source'/BENCHMARK/source.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        assert dest.read_bytes() == raw
    else:
        dest.write_bytes(raw)
    manifest = {
        'authorization': 'Aaron requested the additional Labeling-paper split on 2026-10-13',
        'benchmark': BENCHMARK, 'test_pairs': 4437, 'positive_pairs': 361,
        'source_path': str(source), 'source_sha256': SOURCE_HASH,
        'test_sha256': digest(rows), 'models': MODELS, 'variant': 'B0',
        'additional_decisions': 31059, 'project_cap_usd': 100,
        'paper_source': PAPER, 'historical_f1_percent': HISTORICAL,
        'selection': 'No new development sampling or prompt selection; original AP2 remains frozen',
        'reporting': 'Separate supplement; not a ninth independent benchmark in the original macro average',
    }
    write_frozen(PROJECT/'configs/billiger50_manifest.json', manifest)
    print(canonical(manifest), flush=True)


def verify():
    spec = json.loads((PROJECT/'configs/billiger50_manifest.json').read_text())
    rows = json.loads((PROJECT/'data'/BENCHMARK/'test.json').read_text())
    assert digest(rows) == spec['test_sha256']
    assert len(rows) == 4437 and sum(r['label'] for r in rows) == 361
    assert spec['models'] == MODELS and spec['variant'] == 'B0'


def analyze():
    verify()
    out = PROJECT/'artifacts/results/billiger50'
    out.mkdir(parents=True, exist_ok=True)
    cells = {m: load(m, 'B0', BENCHMARK, 'test') for m in MODELS}
    metrics, intervals, reliability = [], [], []
    for m, data in cells.items():
        metrics.append({'model': m, **metric(data['y'], data['pred']),
                        'known_cost_usd': float(data['costs'].sum()),
                        'unknown_cost_attempts': data['unknown_cost_attempts']})
        if m != 'luna':
            intervals.append({'model': m, 'control': 'luna', **bootstrap_delta(
                data['cases'], data['y'], data['pred'], cells['luna']['pred'])})
            reliability.append({'model': m, **calibration(data['y'], data['prob'])})
    csv_save(out/'metrics.csv', metrics)
    for name, value in [('metrics', metrics), ('paired_intervals', intervals),
                        ('reliability', reliability), ('historical_reference', {
                            'source': PAPER, 'f1_percent': HISTORICAL,
                            'caveat': 'Published single zero-shot runs with different prompts; no paired significance claim'})]:
        (out/(name+'.json')).write_text(canonical(value)+'\n')
    lines = ['# Additional billiger.de comparison on the 50% unseen split', '',
             'All seven configurations use the unchanged B0 request and all 4,437 test pairs '
             '(361 matches). The original 100% unseen split and AP2 selection are unchanged. '
             'These results are reported separately from the original eight-benchmark macro average.', '',
             '| Model | Source | Match F1 (%) | Known API cost (USD) | Unknown-cost attempts |',
             '| --- | --- | ---: | ---: | ---: |']
    for r in metrics:
        lines.append(f"| {r['model']} | Current B0 | {r['f1']*100:.2f} | {r['known_cost_usd']:.6f} | {r['unknown_cost_attempts']} |")
    for m, f1 in HISTORICAL.items():
        lines.append(f'| {m} | Labeling paper | {f1:.2f} | Not measured here | Not measured here |')
    lines.extend(['', f'Historical results: [Labeling paper, Table 7]({PAPER}). '
                  'Identical test-split choice removes the split mismatch; prompts and model versions '
                  'still differ. Historical comparisons are descriptive. Paired intervals are available '
                  'only for current models versus current Luna. Unknown charges are not zero. '
                  'No new human audit or submission readiness is claimed.', ''])
    (out/'comparison.md').write_text('\n'.join(lines))
    print('COMPLETE billiger50 7 cells 31059 predictions', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'run', 'analyze'])
    ap.add_argument('--source', type=Path)
    args = ap.parse_args()
    if args.mode == 'prepare':
        assert args.source is not None
        prepare(args.source)
        return
    verify()
    if args.mode == 'analyze':
        analyze()
        return
    progress = json.loads((PROJECT/'artifacts/progress.json').read_text())
    assert progress['AP3_AP4_analysis']['status'] == 'complete_automated', 'Core must complete first'
    stage = 'supplement_billiger50'
    try:
        stamp(stage, 'running', '7 models x 4437 pairs; unchanged B0; shared $100 ceiling')
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
            futures = [pool.submit(run_cell, m, 'B0', BENCHMARK, 'test', workers=16) for m in MODELS]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        analyze()
        stamp(stage, 'complete_automated', '7 cells and historical context complete; manuscript integration pending')
    except Exception as exc:
        stamp(stage, 'needs_attention', str(exc))
        raise


if __name__ == '__main__':
    main()
