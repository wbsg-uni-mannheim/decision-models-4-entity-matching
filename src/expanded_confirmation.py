"""E2/E3 development selection and full-test confirmation, separate from core AP2."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
import analyze
import run
from expanded_protocol import PROMPTS, payload
from prepare import write_frozen
from protocol import BENCHMARKS, MODELS, canonical, digest

PROJECT = Path(__file__).resolve().parents[1]
PUBLICATION = {'dblp-acm', 'dblp-scholar', 'semi-heter'}
PROMPT_VARIANTS = ['X_P_'+category+'_'+str(i) for category in PROMPTS for i in range(3)]
# S1 has exactly the same payload as X_R_json; use its existing core cache directly.
SERIALIZATIONS = ['B0', 'S1', 'X_R_aligned']
REGIMES = ['global', 'per_benchmark', 'leave_one_out', 'opposite_domain']


def read(path):
    return json.loads(path.read_text())


def sources(target, regime):
    if regime == 'global':
        return list(BENCHMARKS)
    if regime == 'per_benchmark':
        return [target]
    if regime == 'leave_one_out':
        return [b for b in BENCHMARKS if b != target]
    if regime == 'opposite_domain':
        return [b for b in BENCHMARKS if (b in PUBLICATION) != (target in PUBLICATION)]
    raise ValueError(regime)


def choose_variant(scores, variants, source_benchmarks):
    if not source_benchmarks:
        raise ValueError('No source benchmarks; cannot select on target test')
    means = {v: float(np.mean([scores[(b, v)] for b in source_benchmarks])) for v in variants}
    return min(variants, key=lambda v: (-means[v], v)), means


def make_selection(scores):
    """Pure selection: accepts only the complete development score matrix."""
    selected = []
    for m in MODELS:
        per_model = {(b, v): scores[(m, b, v)] for b in BENCHMARKS for v in PROMPT_VARIANTS+SERIALIZATIONS}
        for b in BENCHMARKS:
            for family, variants, regimes in [('prompt', PROMPT_VARIANTS, REGIMES),
                                               ('serialization', SERIALIZATIONS, ['global', 'per_benchmark'])]:
                for regime in regimes:
                    origin = sources(b, regime)
                    best, means = choose_variant(per_model, variants, origin)
                    selected.append({'model': m, 'benchmark': b, 'family': family, 'regime': regime,
                                     'variant': best, 'source_benchmarks': origin,
                                     'development_macro_f1': means, 'selection_split': 'D_prompt',
                                     'test_used_for_selection': False})
    # Additional split is a transfer target, never another source benchmark or macro unit.
    for entry in list(selected):
        if entry['benchmark'] == 'billiger-de':
            selected.append({**entry, 'benchmark': 'billiger-de-50un', 'supplement_transfer_from': 'billiger-de'})
    return selected


def confirmation_cells(selected):
    return sorted({(e['model'], e['variant'], e['benchmark']) for e in selected})


def prepare():
    core = read(PROJECT/'configs/manifest.json')
    supplement = read(PROJECT/'configs/billiger50_manifest.json')
    manifest = {'registration': 'Post-registered after preliminary B0 scores; development-only choices',
                'models': MODELS, 'prompt_variants': PROMPT_VARIANTS, 'serializations': SERIALIZATIONS,
                'regimes': REGIMES, 'serialization_regimes': ['global', 'per_benchmark'],
                'selection': 'Unweighted D_prompt mean positive F1 over declared source benchmarks; lexical variant ID tie break',
                'all_candidates_retained': True, 'screening_does_not_prune_candidates': True,
                'confirmation': 'All unique chosen model/variant/benchmark cells on full frozen tests; shared quality cache',
                'billiger50': 'Transfer only, copied choices from billiger-de family; excluded from original eight-family macro',
                'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'expanded_protocol_sha256': hashlib.sha256((PROJECT/'src/expanded_protocol.py').read_bytes()).hexdigest(),
                'core_manifest_sha256': digest(core), 'supplement_manifest_sha256': digest(supplement),
                'data': {}, 'requests': {}}
    for b in BENCHMARKS:
        manifest['data'][b] = {}
        for split in ['prompt', 'test']:
            cases = read(PROJECT/'data'/b/(split+'.json'))
            assert digest(cases) == core['benchmarks'][b]['splits'][split]['sha256']
            manifest['data'][b][split] = {'n': len(cases), 'sha256': digest(cases)}
        sample = {'benchmark': b, 'records': {'left': {}, 'right': {}}}
        manifest['requests'][b] = {m: {v: payload(sample, m, v) for v in PROMPT_VARIANTS+SERIALIZATIONS} for m in MODELS}
    cases = read(PROJECT/'data/billiger-de-50un/test.json')
    assert len(cases) == 4437 and sum(c['label'] for c in cases) == 361
    assert digest(cases) == supplement['test_sha256']
    manifest['data']['billiger-de-50un'] = {'test': {'n': len(cases), 'sha256': digest(cases)}}
    manifest['maximum_development_decisions_before_cache'] = sum(manifest['data'][b]['prompt']['n'] for b in BENCHMARKS)*len(MODELS)*(len(PROMPT_VARIANTS)+len(SERIALIZATIONS))
    write_frozen(PROJECT/'configs/expanded_confirmation_manifest.json', manifest)
    print('CONFIRMATION PLAN FROZEN; development decisions before cache:', manifest['maximum_development_decisions_before_cache'], flush=True)


def verify():
    manifest = read(PROJECT/'configs/expanded_confirmation_manifest.json')
    assert manifest['implementation_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert manifest['expanded_protocol_sha256'] == hashlib.sha256((PROJECT/'src/expanded_protocol.py').read_bytes()).hexdigest()
    assert manifest['core_manifest_sha256'] == digest(read(PROJECT/'configs/manifest.json'))
    assert manifest['supplement_manifest_sha256'] == digest(read(PROJECT/'configs/billiger50_manifest.json'))
    for b, splits in manifest['data'].items():
        for split, info in splits.items():
            assert digest(read(PROJECT/'data'/b/(split+'.json'))) == info['sha256']
    return manifest


def select():
    scores, rows = {}, []
    for m in MODELS:
        for b in BENCHMARKS:
            for v in PROMPT_VARIANTS+SERIALIZATIONS:
                d = analyze.load(m, v, b, 'prompt')
                metrics = analyze.metric(d['y'], d['pred'])
                scores[(m, b, v)] = metrics['f1']
                rows.append({'model': m, 'benchmark': b, 'variant': v, **metrics,
                             'known_cost_usd': float(d['costs'].sum()), 'unknown_cost_attempts': d['unknown_cost_attempts']})
    selected = make_selection(scores)
    # Freeze ALL regimes together before submitting or reading new test confirmations.
    write_frozen(PROJECT/'configs/expanded_selected_variants.json', selected)
    write_frozen(PROJECT/'configs/expanded_confirmation_cells.json', confirmation_cells(selected))
    out = PROJECT/'artifacts/results/expanded_confirmation'
    out.mkdir(parents=True, exist_ok=True)
    (out/'development_metrics.json').write_text(canonical(rows)+'\n')
    analyze.csv_save(out/'development_metrics.csv', rows)
    return selected


def planned_cost(cells, split, stage):
    """Forecast outstanding calls from corresponding B0 observed input family costs."""
    rows = []
    for m, v, b in cells:
        cases = read(PROJECT/'data'/b/(split+'.json'))
        path = PROJECT/'runs/quality'/m/v/b/(split+'.jsonl')
        cached = {r['pair_id']: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}
        for c in cases:
            if c['pair_id'] in cached:
                assert cached[c['pair_id']]['request_hash'] == digest(payload(c, m, v))
        missing = sum(c['pair_id'] not in cached for c in cases)
        source = 'billiger-de' if b == 'billiger-de-50un' else b
        reference = analyze.load(m, 'B0', source, 'prompt')
        allowance = (float(reference['costs'].sum())+.05*reference['unknown_cost_attempts'])/len(reference['cases'])
        # Prompt changes alter token counts: this is only a baseline-based estimate.
        rows.append({'model': m, 'variant': v, 'benchmark': b, 'split': split,
                     'total_pairs': len(cases), 'cached_pairs': len(cases)-missing, 'new_pairs': missing,
                     'estimated_usd': missing*allowance})
    estimate = {'basis': 'B0 D_prompt observed cost including conservative .05 unknown-attempt allowance; changed prompt lengths may change actual cost',
                'budget_snapshot': read(PROJECT/'artifacts/budget.json'), 'cap_usd': 100,
                'new_pairs': sum(r['new_pairs'] for r in rows), 'estimated_usd': sum(r['estimated_usd'] for r in rows),
                'cells': rows}
    path = PROJECT/'artifacts'/('expanded_'+stage+'_cost_forecast.json')
    if not path.exists():
        path.write_text(canonical(estimate)+'\n')
    print('COST FORECAST', stage, estimate['new_pairs'], 'calls', estimate['estimated_usd'], 'USD; shared hard cap unchanged', flush=True)


def run_cells(cells, split):
    def model_task(m):
        for model, variant, b in cells:
            if model == m:
                run.run_cell(model, variant, b, split, workers=8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        list(pool.map(model_task, MODELS))


def report(selected):
    out = PROJECT/'artifacts/results/expanded_confirmation'
    rows, intervals, costs, macro = [], [], [], []
    cells = {key: analyze.load(key[0], key[1], key[2], 'test') for key in confirmation_cells(selected)}
    for (m, v, b), d in cells.items():
        baseline = analyze.load(m, 'B0', b, 'test')
        intervals.append({'model': m, 'variant': v, 'benchmark': b, 'control': 'same-model B0',
                          **analyze.bootstrap_delta(d['cases'], d['y'], d['pred'], baseline['pred'])})
        costs.append({'model': m, 'variant': v, 'benchmark': b, 'n': len(d['cases']),
                      'known_cost_usd': float(d['costs'].sum()), 'unknown_cost_attempts': d['unknown_cost_attempts']})
    for e in selected:
        d = cells[(e['model'], e['variant'], e['benchmark'])]
        rows.append({k: e[k] for k in ['model', 'benchmark', 'family', 'regime', 'variant']} | analyze.metric(d['y'], d['pred']))
    for m in MODELS:
        for family, regimes in [('prompt', REGIMES), ('serialization', ['global', 'per_benchmark'])]:
            for regime in regimes:
                included = [r for r in rows if r['model'] == m and r['family'] == family and r['regime'] == regime and r['benchmark'] in BENCHMARKS]
                assert len(included) == len(BENCHMARKS)
                macro.append({'model': m, 'family': family, 'regime': regime,
                              'benchmarks': list(BENCHMARKS), 'macro_f1': float(np.mean([r['f1'] for r in included]))})
    for name, value in [('metrics', rows), ('macro', macro), ('intervals', intervals), ('unique_test_cell_costs', costs)]:
        (out/(name+'.json')).write_text(canonical(value)+'\n')
    analyze.csv_save(out/'metrics.csv', rows)
    # Duplicated selected regimes refer to the same cell, not separate billed calls.
    (out/'COST_README.txt').write_text('Cost rows cover unique test cache cells and include retries/unknown attempts. Existing B0 or core S1 cells are reused and already charged in the core; do not add them twice to project costs. Development selection cost is additional and lives in its unique prompt cache cells. The project ledger is authoritative for total accounted spend.\n')
    write_frozen(PROJECT/'artifacts/expanded_confirmation_complete.json',
                 {'status': 'complete', 'selection_sha256': digest(selected), 'unique_test_cells': len(cells),
                  'selected_regime_rows': len(rows), 'full_expanded_study_complete': False})
    print('E2/E3 selection confirmations complete:', len(cells), 'unique full-test cells', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'run', 'analyze'])
    args = ap.parse_args()
    if args.mode == 'prepare':
        prepare()
        return
    verify()
    run.payload = payload
    analyze.payload = payload
    if args.mode == 'run':
        assert read(PROJECT/'artifacts/progress.json')['expanded_E2_E3_screen']['status'] == 'complete'
        dev_cells = [(m, v, b) for m in MODELS for v in PROMPT_VARIANTS+SERIALIZATIONS for b in BENCHMARKS]
        planned_cost(dev_cells, 'prompt', 'confirmation_development')
        run_cells(dev_cells, 'prompt')
    selected = select()
    if args.mode == 'run':
        test_cells = confirmation_cells(selected)
        planned_cost(test_cells, 'test', 'confirmation_test')
        run_cells(test_cells, 'test')
    report(selected)


if __name__ == '__main__':
    main()
