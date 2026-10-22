"""Fixed choice schemas and review abstention with order-preserving wire encoding."""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import threading
from pathlib import Path

import numpy as np
import analyze
import protocol
import run
from prepare import write_frozen
from protocol import BENCHMARKS, MODELS, canonical, digest

PROJECT = Path(__file__).resolve().parents[1]
BINARY = ['C_match', 'C_match_reverse', 'C_same', 'C_same_reverse']
CONDITIONS = [*BINARY, 'C_review']
CONTEXT = threading.local()
ORIGINAL_REQUEST = run.request
DESCRIPTIONS = {'Match': 'Match: The records describe the same real-world entity.',
                'No Match': 'No Match: The records describe different real-world entities.',
                'Same entity': 'Same entity: The records describe the same real-world entity.',
                'Different entities': 'Different entities: The records describe different real-world entities.',
                'Review': 'Review: The available evidence is insufficient to confidently choose Match or No Match; defer this pair for review.'}


def read(path):
    return json.loads(path.read_text())


def options(variant):
    assert variant in CONDITIONS
    labels = ['Same entity', 'Different entities'] if variant.startswith('C_same') else ['Match', 'No Match']
    if variant.endswith('_reverse'):
        labels.reverse()
    if variant == 'C_review':
        labels.append('Review')
    return labels


def payload(case, alias, variant):
    body = protocol.payload(case, alias, 'B0')
    labels = options(variant)
    if MODELS[alias]['kind'] == 'decision':
        body['questions']['match'] = {'type': 'choice', 'instructions': protocol.BASE,
                                      'criteria': {s: DESCRIPTIONS[s] for s in labels}}
    else:
        body['messages'][0]['content'] = protocol.BASE + '\nAllowed answers:\n' + '\n'.join(DESCRIPTIONS[s] for s in labels)
        body['response_format']['json_schema']['schema']['properties']['match'] = {'type': 'string', 'enum': labels}
    return body


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def wire_hash(value):
    return hashlib.sha256(wire(value).encode()).hexdigest()


def parse(alias, data, variant):
    labels = options(variant)
    if MODELS[alias]['kind'] == 'decision':
        answer = data['answers']['match']
        probs = answer['probabilities']
        if answer['type'] != 'choice' or set(probs) != set(labels):
            raise ValueError('Unexpected choice schema')
        if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()):
            raise ValueError('Invalid choice probability')
        if abs(sum(probs.values())-1) > .001:
            raise ValueError('Choice probabilities do not sum to one')
        selected = answer['choice']
        if selected not in labels or probs[selected] < max(probs.values())-1e-6:
            raise ValueError('Choice is not an argmax')
        positive = 'Same entity' if variant.startswith('C_same') else 'Match'
        probability = float(probs[positive])
        # Reuse frozen provider/version validation, without changing the original parser.
        check = copy.deepcopy(data)
        check['answers']['match'] = {'type': 'noul', 'noul': probability}
        protocol.parse_response(alias, check)
    else:
        choice, = data['choices']
        obj = json.loads(choice['message']['content'])
        if set(obj) != {'match'} or obj['match'] not in labels:
            raise ValueError('Unexpected generative choice schema')
        selected, probability = obj['match'], None
        check = copy.deepcopy(data)
        check['choices'][0]['message']['content'] = json.dumps({'match': selected in ['Match', 'Same entity']})
        protocol.parse_response(alias, check)
    return (-1 if selected == 'Review' else int(selected in ['Match', 'Same entity'])), probability


def request(case, alias, variant, credential):
    CONTEXT.variant = variant
    return ORIGINAL_REQUEST(case, alias, variant, credential)


def install():
    # Only this process changes; core files, core serialization and core request hashes stay untouched.
    run.payload = payload
    run.canonical = wire
    run.digest = wire_hash
    run.parse_response = lambda alias, data: parse(alias, data, CONTEXT.variant)
    run.request = request


def prepare():
    manifest = {'registration': 'Post-registered; selection only on D_prompt, never test',
                'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'protocol_sha256': hashlib.sha256((PROJECT/'src/protocol.py').read_bytes()).hexdigest(),
                'models': MODELS, 'conditions': {v: options(v) for v in CONDITIONS},
                'descriptions': DESCRIPTIONS, 'pilot': {'families': ['abt-buy', 'dblp-scholar'], 'pairs_per_family': 2, 'max_new_calls_before_retries': 140,
                   'rule': 'Every model/condition must pass; failures retained and block continuation. No automatic model exclusion.'},
                'development': 'All five conditions on D_prompt of all eight families; no new sample selection',
                'selection': 'Binary choices: global and per-benchmark match F1; lexical variant tie; freeze before test. Review always confirmed.',
                'abstention': 'Review=-1; report coverage, accepted risk/F1, review class mix and resolved correct/all. Never score Review as a third gold class.',
                'wire_order': 'Preserve criteria insertion order and generative enum/instruction order; SHA256 exact JSON bytes. Provider may reorder upstream; not claimed as verified internal order.',
                'confounds': 'Choice key names may not reach decision models; descriptions explicitly include labels. Schema and answer wording change together; no pure architecture effect.',
                'api_reference': 'https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request',
                'project_cap_usd': 100, 'benchmarks': {}, 'new_development_decisions_before_cache': 5*7*8*200,
                'confirmation_upper_bound_decisions': 3*7*(30225+4437)}
    for b in BENCHMARKS:
        rows = read(PROJECT/'data'/b/'prompt.json')
        assert len(rows) == 200
        manifest['benchmarks'][b] = {'development_sha256': digest(rows)}
        if b in manifest['pilot']['families']:
            pilot = sorted(rows, key=lambda c: digest(['schema-pilot-v1', b, c['pair_id']]))[:2]
            write_frozen(PROJECT/'data'/b/'schema_pilot.json', pilot)
            manifest['benchmarks'][b]['pilot_sha256'] = digest(pilot)
    sample = {'benchmark': 'wdc', 'records': {'left': {}, 'right': {}}}
    manifest['wire_templates'] = {m: {v: wire(payload(sample, m, v)) for v in CONDITIONS} for m in MODELS}
    write_frozen(PROJECT/'configs/expanded_schema_manifest.json', manifest)


def verify():
    spec = read(PROJECT/'configs/expanded_schema_manifest.json')
    assert spec['implementation_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert spec['protocol_sha256'] == hashlib.sha256((PROJECT/'src/protocol.py').read_bytes()).hexdigest()
    assert spec['models'] == MODELS
    for b, info in spec['benchmarks'].items():
        assert digest(read(PROJECT/'data'/b/'prompt.json')) == info['development_sha256']
        if 'pilot_sha256' in info:
            assert digest(read(PROJECT/'data'/b/'schema_pilot.json')) == info['pilot_sha256']
    return spec


def load(m, v, b, split, namespace='schema'):
    cases = read(PROJECT/'data'/b/(split+'.json'))
    path = PROJECT/'runs'/namespace/m/v/b/(split+'.jsonl')
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    indexed = {r['pair_id']: r for r in rows}
    assert len(indexed) == len(rows) and indexed.keys() == {c['pair_id'] for c in cases}
    ordered = [indexed[c['pair_id']] for c in cases]
    for c, r in zip(cases, ordered):
        assert r['request_hash'] == wire_hash(payload(c, m, v))
        parsed = parse(m, r['attempts'][-1]['response'], v)
        assert parsed == (r['prediction'], r['probability_match'])
    all_rows = rows[:]
    failed = path.with_name(split+'.failures.jsonl')
    if failed.exists():
        failures = [json.loads(x) for x in failed.read_text().splitlines()]
        for r in failures:
            assert r['pair_id'] in indexed and r['request_hash'] == indexed[r['pair_id']]['request_hash']
        all_rows.extend(failures)
    attempts = [a for r in all_rows for a in r['attempts']]
    return {'cases': cases, 'y': np.array([c['label'] for c in cases]), 'pred': np.array([r['prediction'] for r in ordered]),
            'known_cost_usd': sum(a['cost'] for a in attempts if a['cost'] is not None),
            'unknown_cost_attempts': sum(a['cost'] is None for a in attempts)}


def selective_metrics(y, pred):
    accepted = pred != -1
    return {'n_total': len(y), 'coverage': float(accepted.mean()), 'review_count': int((~accepted).sum()),
            'review_positive': int(y[~accepted].sum()), 'accepted_risk': float((pred[accepted] != y[accepted]).mean()) if accepted.any() else None,
            'resolved_correct_over_all': float(((pred == y) & accepted).mean()),
            'accepted_metrics': analyze.metric(y[accepted], pred[accepted])}


def pilot():
    spec = verify()
    rows = []
    for m in MODELS:
        for v in CONDITIONS:
            for b in spec['pilot']['families']:
                try:
                    run.run_cell(m, v, b, 'schema_pilot', workers=2, namespace='schema_pilot')
                    d = load(m, v, b, 'schema_pilot', 'schema_pilot')
                    rows.append({'model': m, 'condition': v, 'benchmark': b, 'status': 'passed',
                                 'known_cost_usd': d['known_cost_usd'], 'unknown_cost_attempts': d['unknown_cost_attempts']})
                except Exception as exc:
                    rows.append({'model': m, 'condition': v, 'benchmark': b, 'status': 'failed_requires_diagnosis', 'error': str(exc)})
                    (PROJECT/'artifacts/schema_pilot_status.json').write_text(canonical({'complete': False, 'cells': rows})+'\n')
                    raise
                (PROJECT/'artifacts/schema_pilot_status.json').write_text(canonical({'complete': False, 'cells': rows})+'\n')
    (PROJECT/'artifacts/schema_pilot_status.json').write_text(canonical({'complete': True, 'cells': rows})+'\n')


def select():
    selected, cells = {}, set()
    for m in MODELS:
        scores = {v: {b: analyze.metric(d['y'], d['pred'])['f1'] for b in BENCHMARKS for d in [load(m, v, b, 'prompt')]} for v in BINARY}
        global_choice = min(BINARY, key=lambda v: (-np.mean(list(scores[v].values())), v))
        local = {b: min(BINARY, key=lambda v: (-scores[v][b], v)) for b in BENCHMARKS}
        local['billiger-de-50un'] = local['billiger-de']
        selected[m] = {'development_scores': scores, 'global': global_choice, 'per_benchmark': local}
        for b in [*BENCHMARKS, 'billiger-de-50un']:
            for v in {global_choice, local[b], 'C_review'}:
                cells.add((m, v, b))
    write_frozen(PROJECT/'configs/expanded_schema_selection.json', selected)
    write_frozen(PROJECT/'configs/expanded_schema_test_cells.json', sorted(cells))
    return sorted(cells)


def forecast(cells, phase):
    rows = []
    for m, v, b, split in cells:
        family = 'billiger-de' if b == 'billiger-de-50un' else b
        source = analyze.load(m, 'B0', family, 'prompt')
        n = len(read(PROJECT/'data'/b/(split+'.json')))
        rows.append({'model': m, 'condition': v, 'benchmark': b, 'split': split, 'n': n,
                     'estimated_usd': n*(float(source['costs'].sum())+.05*source['unknown_cost_attempts'])/len(source['cases'])})
    value = {'basis': 'B0 development costs incl. errors and unknown-cost allowance .05; schemas change costs, gross estimate before cache',
             'rows': rows, 'estimated_usd': sum(r['estimated_usd'] for r in rows), 'budget_snapshot': read(PROJECT/'artifacts/budget.json')}
    path = PROJECT/'artifacts'/('schema_'+phase+'_cost_forecast.json')
    if not path.exists():
        path.write_text(canonical(value)+'\n')
    print('SCHEMA', phase, 'estimated gross USD', value['estimated_usd'], flush=True)


def report():
    selected = read(PROJECT/'configs/expanded_schema_selection.json')
    cells = read(PROJECT/'configs/expanded_schema_test_cells.json')
    metrics, intervals, costs, development, order_effects = [], [], [], [], []
    for m, v, b in cells:
        d = load(m, v, b, 'test')
        baseline = analyze.load(m, 'B0', b, 'test')
        assert [c['pair_id'] for c in d['cases']] == [c['pair_id'] for c in baseline['cases']]
        accepted = d['pred'] != -1
        metrics.append({'model': m, 'condition': v, 'benchmark': b, **selective_metrics(d['y'], d['pred']),
                        'baseline_same_accepted_subset': analyze.metric(baseline['y'][accepted], baseline['pred'][accepted]),
                        'global_selected': v == selected[m]['global'], 'local_selected': v == selected[m]['per_benchmark'][b]})
        if v != 'C_review':
            intervals.append({'model': m, 'condition': v, 'benchmark': b,
                              **analyze.bootstrap_delta(d['cases'], d['y'], d['pred'], baseline['pred'])})
        costs.append({'model': m, 'condition': v, 'benchmark': b, 'split': 'test',
                      'known_cost_usd': d['known_cost_usd'], 'unknown_cost_attempts': d['unknown_cost_attempts']})
    for m in MODELS:
        for v in CONDITIONS:
            for b in BENCHMARKS:
                d = load(m, v, b, 'prompt')
                development.append({'model': m, 'condition': v, 'benchmark': b, **selective_metrics(d['y'], d['pred'])})
                costs.append({'model': m, 'condition': v, 'benchmark': b, 'split': 'prompt',
                              'known_cost_usd': d['known_cost_usd'], 'unknown_cost_attempts': d['unknown_cost_attempts']})
    for m in MODELS:
        for b in BENCHMARKS:
            for v in ['C_match', 'C_same']:
                a, z = load(m, v, b, 'prompt'), load(m, v+'_reverse', b, 'prompt')
                assert [c['pair_id'] for c in a['cases']] == [c['pair_id'] for c in z['cases']]
                order_effects.append({'model': m, 'benchmark': b, 'condition': v,
                                     'development_flip_rate': float((a['pred'] != z['pred']).mean()),
                                     'reverse_minus_forward_f1': analyze.metric(z['y'], z['pred'])['f1']-analyze.metric(a['y'], a['pred'])['f1']})
    macro = []
    for m in MODELS:
        for regime in ['global_selected', 'local_selected']:
            rows = [r for r in metrics if r['model'] == m and r['benchmark'] in BENCHMARKS and r[regime]]
            assert len(rows) == len(BENCHMARKS)
            macro.append({'model': m, 'regime': regime, 'benchmarks': len(rows),
                          'macro_match_f1': float(np.mean([r['accepted_metrics']['f1'] for r in rows]))})
    out = PROJECT/'artifacts/results/expanded_schema';out.mkdir(parents=True, exist_ok=True)
    for name, value in [('metrics', metrics), ('development', development), ('order_effects_development', order_effects), ('macro', macro), ('intervals', intervals), ('costs', costs), ('pilot', read(PROJECT/'artifacts/schema_pilot_status.json'))]:
        (out/(name+'.json')).write_text(canonical(value)+'\n')
    write_frozen(PROJECT/'artifacts/expanded_schema_complete.json', {'status': 'complete', 'test_cells': len(cells), 'full_expanded_study_complete': False})


def main():
    ap = argparse.ArgumentParser();ap.add_argument('mode', choices=['prepare', 'pilot', 'run', 'analyze']);args = ap.parse_args()
    if args.mode == 'prepare':
        prepare();return
    verify();install()
    if args.mode in ['pilot', 'run']:
        pilot()
        if args.mode == 'pilot':
            return
    if args.mode == 'run':
        dev = [(m, v, b, 'prompt') for m in MODELS for v in CONDITIONS for b in BENCHMARKS]
        forecast(dev, 'development')
        def dev_model(m):
            for _, v, b, split in [x for x in dev if x[0] == m]:
                run.run_cell(m, v, b, split, workers=8, namespace='schema')
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
            list(pool.map(dev_model, MODELS))
        cells = select()
        forecast([(*c, 'test') for c in cells], 'confirmation')
        def test_model(m):
            for _, v, b in [x for x in cells if x[0] == m]:
                run.run_cell(m, v, b, 'test', workers=8, namespace='schema')
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
            list(pool.map(test_model, MODELS))
    report()


if __name__ == '__main__':
    main()
