"""Freeze and run the broad development screening after the existing jobs."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import run
import analyze
from protocol import MODELS, canonical, digest
from prepare import case, choose, read_rows, write_frozen
from orchestrate import stamp
from expanded_protocol import VARIANTS, PROMPTS, ALIASES, payload

PROJECT = Path(__file__).resolve().parents[1]
SCREEN_BENCHMARKS = ['abt-buy', 'wdc', 'dblp-scholar']


def freeze():
    manifest = {'authorization': 'Aaron 2026-10-13: all investigations, existing models and benchmark families',
                'registration': 'Added after initial B0 test results were seen; exploratory extension, never selected on test scores',
                'models': MODELS, 'variants': VARIANTS, 'prompts': PROMPTS, 'aliases': ALIASES,
                'benchmarks': {}, 'stage': 'E2_E3_development_screening', 'pairs_per_benchmark': 500,
                'requests': 7*3*500*len(VARIANTS), 'cap_usd_shared_with_core': 100,
                'information_ablation': ['X_R_values', 'X_R_aliases', 'X_R_missing'],
                'preprocessing_ablation': ['X_R_normalized'], 'selection': 'No test inference or winner selection in this stage'}
    core = json.loads((PROJECT/'configs/manifest.json').read_text())
    for b in SCREEN_BENCHMARKS:
        directory = PROJECT/'data/source'/b
        valid = directory/('wdcproducts80cc20rnd000un_valid_large.json.gz' if b == 'wdc' else b+'-valid.csv')
        rows = read_rows(valid)
        if set(rows[0]) == {'pair_id'}:
            source = directory/(b+'-valid.json.gz')
            if not source.exists():
                source = directory/(b+'-train.json')
            indexed = {str(r['pair_id']): r for r in read_rows(source)}
            rows = [indexed[str(r['pair_id'])] for r in rows]
        rows = [case(r, b) for r in rows]
        excluded = set()
        for split in ['test', 'prompt', 'route']:
            existing = json.loads((PROJECT/'data'/b/(split+'.json')).read_text())
            excluded.update(r['input_hash'] for r in existing)
        unique, ambiguous = {}, set()
        for r in rows:
            h = r['input_hash']
            if h in excluded:
                continue
            if h in unique and unique[h]['label'] != r['label']:
                ambiguous.add(h)
            unique.setdefault(h, r)
        eligible = [r for h, r in unique.items() if h not in ambiguous]
        assert len(eligible) >= 500
        selected = choose(eligible, 500, '20261013:expanded-screen:'+b)
        assert not {r['input_hash'] for r in selected} & excluded
        write_frozen(PROJECT/'data'/b/'expanded_screen.json', selected)
        manifest['benchmarks'][b] = {'n': len(selected), 'positive': sum(r['label'] for r in selected),
                                     'sha256': digest(selected), 'source_sha256': core['benchmarks'][b]['source_sha256']}
    example = {'benchmark': 'wdc', 'records': {'left': {}, 'right': {}}}
    manifest['request_templates'] = {m: {v: payload(example, m, v) for v in VARIANTS} for m in MODELS}
    write_frozen(PROJECT/'configs/expanded_screen_manifest.json', manifest)
    print('FROZEN SCREEN', manifest['requests'], flush=True)


def verify():
    manifest = json.loads((PROJECT/'configs/expanded_screen_manifest.json').read_text())
    assert manifest['variants'] == VARIANTS and manifest['models'] == MODELS
    for b, info in manifest['benchmarks'].items():
        rows = json.loads((PROJECT/'data'/b/'expanded_screen.json').read_text())
        assert digest(rows) == info['sha256']
    sample = {'benchmark': 'wdc', 'records': {'left': {}, 'right': {}}}
    assert manifest['request_templates'] == {m: {v: payload(sample, m, v) for v in VARIANTS} for m in MODELS}


def model_task(model):
    for variant in VARIANTS:
        for b in SCREEN_BENCHMARKS:
            run.run_cell(model, variant, b, 'expanded_screen', workers=8)


def report():
    out = PROJECT/'artifacts/results/expanded_screen'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in MODELS:
        for b in SCREEN_BENCHMARKS:
            for v in VARIANTS:
                d = analyze.load(m, v, b, 'expanded_screen')
                rows.append({'model': m, 'benchmark': b, 'variant': v,
                             **analyze.metric(d['y'], d['pred']),
                             'known_cost_usd': float(d['costs'].sum()),
                             'unknown_cost_attempts': d['unknown_cost_attempts']})
    analyze.csv_save(out/'development_metrics.csv', rows)
    (out/'development_metrics.json').write_text(canonical(rows)+'\n')
    print('DEVELOPMENT SCREEN COMPLETE', len(rows), 'cells; full expanded plan remains pending', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'run', 'analyze'])
    args = ap.parse_args()
    if args.mode == 'prepare':
        freeze()
        return
    verify()
    run.payload = payload
    analyze.payload = payload
    if args.mode == 'analyze':
        report()
        return
    progress = json.loads((PROJECT/'artifacts/progress.json').read_text())
    assert progress['supplement_billiger50']['status'] == 'complete_automated'
    stage = 'expanded_E2_E3_screen'
    try:
        stamp(stage, 'running', '27 conditions x 7 models x 3 development benchmarks x 500 pairs')
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
            futures = [pool.submit(model_task, m) for m in MODELS]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        report()
        stamp(stage, 'complete', 'Screening only; see expanded workplan for remaining experiments and confirmations')
    except Exception as exc:
        stamp(stage, 'needs_attention', str(exc))
        raise


if __name__ == '__main__':
    main()
