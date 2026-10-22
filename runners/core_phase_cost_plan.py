"""Read-only historical scenarios for every remaining initial fresh-core phase.

AP2 choices are unknown until each fresh run's development finishes. Price all
three eligible choices from their own development-cell means, then take the
largest model-wide scenario. This is not a provider or token-length bound.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import sys

from diagnose_confirmation_terminal import audit_confirmation_rows
from repeated_runs import verify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def aggregate(rows):
    return {'known_proxy_usd': math.fsum(r['known_proxy_usd'] for r in rows),
            'historical_unknown_scenario_usd': math.fsum(r['historical_unknown_scenario_usd'] for r in rows)}


def costs(attempts):
    values = [a['cost'] for a in attempts if a['cost'] is not None]
    if any(type(c) not in (int, float) or not math.isfinite(c) or c < 0 for c in values):
        raise ValueError('Invalid historical cost')
    return {'known_proxy_usd': math.fsum(values),
            'historical_unknown_scenario_usd': .05*sum(a['cost'] is None for a in attempts)}


def confirmation_scenarios(sources, models, benchmarks, test_counts):
    candidates, envelopes = [], []
    for model in models:
        model_candidates = []
        for variant in ('P1', 'P2', 'S1'):
            cells = []
            for benchmark in benchmarks:
                source = sources['quality', model, variant, benchmark, 'prompt']
                if source['responses'] <= 0:
                    raise ValueError('Empty development cost source')
                n = test_counts[benchmark]
                scale = n/source['responses']
                cells.append({'benchmark': benchmark, 'prospective_responses': n,
                    'source_path': source['path'], 'source_responses': source['responses'],
                    **{k: source[k]*scale for k in
                       ('known_proxy_usd', 'historical_unknown_scenario_usd')}})
            candidate = {'model': model, 'variant': variant, 'cells': cells, **aggregate(cells)}
            model_candidates.append(candidate)
        # Separate maxima make both components at least as large as any one
        # historical scenario; they need not correspond to the same variant.
        envelopes.append({'model': model, **{k: max(c[k] for c in model_candidates)
            for k in ('known_proxy_usd', 'historical_unknown_scenario_usd')}})
        candidates.extend(model_candidates)
    return candidates, envelopes


def build(root, output):
    if output.exists():
        raise FileExistsError(output)
    def denied(*args, **kwargs):
        raise AssertionError('Network prohibited in cost planning')
    sys.path.insert(0, str(root/'src'))
    import protocol
    import orchestrate
    socket.socket = denied
    if orchestrate.PROJECT != root:
        raise ValueError('Wrong source workspace')
    orchestrate.verify()
    specs = {i: verify(root, i) for i in (2, 3)}
    for rel, expected in specs[2]['immutable_sha256'].items():
        if rel.startswith('src/') and sha(root/rel) != expected:
            raise ValueError('Run1 frozen source drift: '+rel)
    protected = ['artifacts/budget.json', 'artifacts/progress.json', 'configs/selected_variants.json']
    protected += [f'replications/run{i:02d}/artifacts/{name}' for i in (2, 3)
                  for name in ('progress.json', 'stage_execution.jsonl')]
    protected_hashes = {p: sha(root/p) for p in protected}
    frozen_paths = [p for p in specs[2]['immutable_sha256'] if p.startswith('src/')]
    frozen_paths += ['configs/manifest.json']
    frozen_paths += [f'data/{b}/{s}.json' for b in protocol.BENCHMARKS for s in ('prompt', 'route', 'test')]
    frozen_hashes = {p: sha(root/p) for p in frozen_paths}
    models, benchmarks = list(protocol.MODELS), list(protocol.BENCHMARKS)
    ablations = list(protocol.ABLATION_MODELS)
    matrix = [('quality', m, 'B0', b, s) for m in models for b in benchmarks
              for s in ('prompt', 'route', 'test')]
    matrix += [('quality', m, v, b, 'prompt') for m in ablations
               for v in ('P1', 'P2', 'S1', 'O1') for b in benchmarks]
    matrix += [(ns, m, 'B0', b, 'prompt') for ns in ('timing_serial', 'timing_concurrent8')
               for m in models for b in benchmarks]
    sources, raw, files = {}, {}, {}
    for ns, m, v, b, split in matrix:
        rel = f'runs/{ns}/{m}/{v}/{b}/{split}.jsonl'
        path = root/rel
        fail = path.with_name(split+'.failures.jsonl')
        successes = [json.loads(x) for x in path.read_text().splitlines()]
        failures = [json.loads(x) for x in fail.read_text().splitlines()] if fail.exists() else []
        cases = read(root/'data'/b/(split+'.json'))
        if ns.startswith('timing_'):
            cases = cases[:20]
        audit = audit_confirmation_rows(cases, successes, failures, protocol, protocol, m, v, b)
        if not audit['complete']:
            raise ValueError('Incomplete historical source')
        amounts = costs([a for r in successes+failures for a in r['attempts']])
        source = {'namespace': ns, 'model': m, 'variant': v, 'benchmark': b, 'split': split,
                  'path': rel, 'responses': len(successes), **amounts, 'audit': audit,
                  'cache_sha256': sha(path), 'failure_sha256': sha(fail) if fail.exists() else None}
        sources[ns, m, v, b, split] = source
        raw[ns, m, v, b, split] = successes+failures
        files[rel] = source['cache_sha256']
        if fail.exists():
            files[str(fail.relative_to(root))] = source['failure_sha256']
    test_counts = {b: len(read(root/'data'/b/'test.json')) for b in benchmarks}
    candidates, envelopes = confirmation_scenarios(sources, ablations, benchmarks, test_counts)
    development = [s for s in sources.values() if s['namespace'] == 'quality' and s['variant'] != 'B0']
    timing = [s for s in sources.values() if s['namespace'].startswith('timing_')]
    plans = {}
    for run_id, diagnosis_rel in ((2, 'run02_terminal_diagnosis_20261015T0103Z.json'),
                                   (3, 'run03_terminal_diagnosis_20261014T2333Z.json')):
        target = root/'replications'/f'run{run_id:02d}'
        dp = root/'artifacts/operations'/diagnosis_rel
        diagnosis = read(dp)
        observed = {str(p.relative_to(target)): sha(p) for p in (target/'runs').rglob('*.jsonl')}
        if (observed != diagnosis['response_files_sha256'] or diagnosis['run_id'] != run_id
                or sha(target/'artifacts/stage_execution.jsonl') != diagnosis['stage_journal_sha256']
                or read(target/'artifacts/progress.json') != diagnosis['progress']):
            raise ValueError('Terminal diagnosis no longer current')
        baseline = []
        for cell in diagnosis['cells']:
            m, b, split = (cell[k] for k in ('model', 'benchmark', 'split'))
            key = ('quality', m, 'B0', b, split)
            cases = read(target/'data'/b/(split+'.json'))
            if cases != read(root/'data'/b/(split+'.json')):
                raise ValueError('Replica/source input mismatch')
            path = target/sources[key]['path']
            current = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
            missing = {c['pair_id'] for c in cases}-{r['pair_id'] for r in current}
            if len(missing) != cell['missing'] or len(current) != cell['responses']:
                raise ValueError('Diagnosis count mismatch')
            attempts = [a for r in raw[key] if r['pair_id'] in missing for a in r['attempts']]
            baseline.append({'model': m, 'benchmark': b, 'split': split, 'source_path': sources[key]['path'],
                'missing': len(missing), 'missing_ids_sha256': protocol.digest(sorted(missing)), **costs(attempts)})
        if len(baseline) != len(models)*len(benchmarks)*3:
            raise ValueError('Incomplete baseline matrix')
        phases = {'AP1_baseline': aggregate(baseline), 'AP2_development': aggregate(development),
                  'AP2_confirmation': aggregate(envelopes), 'AP1_timing': aggregate(timing)}
        plans[str(run_id)] = {'run_id': run_id, 'diagnosis_sha256': sha(dp), 'phases': phases,
            'baseline_missing_cells': baseline, 'total': aggregate(list(phases.values())),
            'quote_or_hard_bound': False, 'execution_admitted': False}
    if any(sha(root/p) != h for p, h in {**protected_hashes, **files, **frozen_hashes}.items()):
        raise ValueError('Source changed while planning')
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'historical_phase_scenarios_not_admission',
        'sources': list(sources.values()), 'source_files_sha256': files, 'protected_sha256': protected_hashes,
        'frozen_source_sha256': frozen_hashes,
        'script_sha256': sha(Path(__file__)), 'plans': plans, 'test_counts': test_counts,
        'confirmation_candidates': candidates, 'confirmation_model_envelopes': envelopes,
        'fresh_selection_known': False, 'new_paid_calls': 0, 'execution_admitted': False,
        'budget_snapshot': read(root/'artifacts/budget.json'),
        'limitations': ['All four inference phases covered as historical scenarios, not a provider quote or hard bound.',
            'Baseline uses same missing pairs and requests; historical unjournaled fresh outcomes can be repeated.',
            'AP2 confirmation uses each candidate development mean scaled to test count, with separate per-model maxima.',
            'Prompt/test token lengths and outputs can differ; maxima bound only these scenarios, not actual future costs.',
            'No test quality chooses the cost scenario; actual fresh development alone will select P1/P2/S1.',
            'Known costs include preserved historical retries; unknown allowances are not fees or retry forecasts.',
            'Timing uses separate original serial/concurrent8 journals, 20 pairs per cell; no quality cache double count.',
            'Cache-only analysis has no inference calls; scheduler compute fees are not estimated.',
            'Excludes supplement, all extensions, selective ensemble, Astra and remaining Run1 work.',
            'Independent source verification and fresh admission/budget/scheduler checks still required.']}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = build(args.root.resolve(), args.output)
    print(json.dumps({k: {'phases': p['phases'], 'total': p['total']} for k, p in report['plans'].items()}))
