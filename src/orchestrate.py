"""Execute the four authorized work packages in an auditable sequence."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from protocol import MODELS, BENCHMARKS, ABLATION_MODELS, canonical, digest
from run import PROJECT, run_cell

def stamp(stage, status, detail=None):
    path = PROJECT/'artifacts/progress.json'
    value = json.loads(path.read_text()) if path.exists() else {}
    value[stage] = {'status': status, 'utc': datetime.now(timezone.utc).isoformat(), 'detail': detail}
    tmp = path.with_suffix('.tmp')
    tmp.write_text(canonical(value)+'\n')
    tmp.replace(path)
    print('STAGE', stage, status, detail, flush=True)

def task(alias, variants, splits, limit=None):
    for variant in variants:
        for split in splits:
            for b in BENCHMARKS:
                run_cell(alias, variant, b, split, workers=8 if limit else 16, limit=limit)

def parallel(aliases, variants, splits, limit=None):
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(aliases)) as pool:
        futures = {pool.submit(task, alias, variants, splits, limit): alias for alias in aliases}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append({'model': futures[future], 'error': str(exc)})
    if failures:
        raise RuntimeError(canonical(failures))

def score(y, pred):
    tp = sum(a == b == 1 for a,b in zip(y,pred))
    fp = sum(a == 0 and b == 1 for a,b in zip(y,pred))
    fn = sum(a == 1 and b == 0 for a,b in zip(y,pred))
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.

def select():
    result = {}
    for m in ABLATION_MODELS:
        values = {}
        for variant in ['P1', 'P2', 'S1']:
            scores = []
            for b in BENCHMARKS:
                cases = json.loads((PROJECT/'data'/b/'prompt.json').read_text())
                path = PROJECT/'runs'/'quality'/m/variant/b/'prompt.jsonl'
                rows = {x['pair_id']: x for x in map(json.loads, path.read_text().splitlines())}
                assert len(rows) == len(cases)
                scores.append(score([x['label'] for x in cases], [rows[x['pair_id']]['prediction'] for x in cases]))
            values[variant] = sum(scores)/len(scores)
        best = max(['P1','P2','S1'], key=lambda v: values[v])
        result[m] = {'variant': best, 'development_macro_match_f1': values,
                     'selection_only': 'D_prompt', 'tie_order': ['P1','P2','S1']}
    path = PROJECT/'configs/selected_variants.json'
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        path.write_text(canonical(result)+'\n')
    return result

def verify():
    manifest = json.loads((PROJECT/'configs/manifest.json').read_text())
    for b, spec in manifest['benchmarks'].items():
        for split, info in spec['splits'].items():
            rows = json.loads((PROJECT/'data'/b/(split+'.json')).read_text())
            assert digest(rows) == info['sha256'], ('Frozen dataset changed', b, split)
    return manifest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['smoke', 'all', 'analyze'])
    args = ap.parse_args()
    verify()
    stage = args.mode
    try:
        if args.mode == 'smoke':
            stamp('smoke', 'running')
            parallel(list(MODELS), ['B0'], ['prompt'], limit=20)
            stamp('smoke', 'complete', '7 configurations x 8 benchmarks x 20 development pairs')
            return
        if args.mode == 'all':
            status = json.loads((PROJECT/'artifacts/progress.json').read_text())
            assert status['smoke']['status'] == 'complete', 'Technical smoke test required'
            stage = 'AP1_baseline'
            stamp(stage, 'running')
            parallel(list(MODELS), ['B0'], ['prompt','route','test'])
            stamp(stage, 'complete')
            stage = 'AP2_development'
            stamp(stage, 'running')
            parallel(ABLATION_MODELS, ['P1','P2','S1','O1'], ['prompt'])
            selection = select()
            stamp(stage, 'complete', selection)
            stage = 'AP2_confirmation'
            stamp(stage, 'running')
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(task, m, [v['variant']], ['test']) for m,v in selection.items()]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            stamp(stage, 'complete')
            stage = 'AP1_timing'
            stamp(stage, 'running')
            # Separate timing invocations; no quality-cache hits. Repeat under serial and concurrent load.
            for m in MODELS:
                for b in BENCHMARKS:
                    for concurrency, name in [(1, 'serial'), (8, 'concurrent8')]:
                        run_cell(m, 'B0', b, 'prompt', workers=concurrency, limit=20, namespace='timing_'+name)
            stamp(stage, 'complete')
        stage = 'AP3_AP4_analysis'
        stamp(stage, 'running')
        subprocess.run([sys.executable, str(PROJECT/'src/analyze.py')], check=True)
        stamp(stage, 'complete_automated', 'Independent human audit and author review remain pending')
    except Exception as exc:
        stamp(stage, 'needs_attention', str(exc))
        raise

if __name__ == '__main__':
    main()
