"""Read-only choice-schema adapters; exact wire hashes, abstention and run provenance."""
import hashlib
import json
import numpy as np


def read(path):
    return json.loads(path.read_text())


def selected_matrix(module, development, metric):
    selected, cells = {}, set()
    for model in module.MODELS:
        scores = {v: {b: metric(development[model, v, b]['y'], development[model, v, b]['pred'])['f1']
                      for b in module.BENCHMARKS} for v in module.BINARY}
        global_choice = min(module.BINARY, key=lambda v: (-np.mean(list(scores[v].values())), v))
        local = {b: min(module.BINARY, key=lambda v: (-scores[v][b], v)) for b in module.BENCHMARKS}
        local['billiger-de-50un'] = local['billiger-de']
        selected[model] = {'development_scores': scores, 'global': global_choice, 'per_benchmark': local}
        for b in [*module.BENCHMARKS, 'billiger-de-50un']:
            cells.update((model, v, b) for v in {global_choice, local[b], 'C_review'})
    return selected, [list(c) for c in sorted(cells)]


def schema_cell(workspace, module, model, variant, benchmark, split, namespace, origin, provenance):
    # Frozen loader validates all exact order-preserving wire hashes, parses raw
    # responses again, and includes failed attempts in known/unknown costs.
    data = module.load(model, variant, benchmark, split, namespace)
    path = workspace/'runs'/namespace/model/variant/benchmark/(split+'.jsonl')
    responses = [json.loads(line) for line in path.read_text().splitlines()]
    failure = path.with_name(split+'.failures.jsonl')
    failures = [json.loads(line) for line in failure.read_text().splitlines()] if failure.exists() else []
    cases = {c['pair_id']: c for c in data['cases']}
    origin(responses+failures)
    for row in responses+failures:
        case = cases[row['pair_id']]
        if row['input_hash'] != case['input_hash']:
            raise ValueError('Schema input provenance drift')
    provenance.append({'model': model, 'variant': variant, 'benchmark': benchmark, 'split': split,
                       'namespace': namespace, 'cache_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'failure_sha256': hashlib.sha256(failure.read_bytes()).hexdigest() if failure.exists() else None,
                       'request_hashes_verified': True, 'wire_order_verified': True,
                       'response_count': len(responses)})
    return data


def condition_rows(run_id, model, benchmark, condition, variant, data, baseline, metric):
    if data['cases'] != baseline['cases'] or not np.array_equal(data['y'], baseline['y']):
        raise ValueError('Schema/B0 test alignment differs')
    y, pred = data['y'], data['pred']
    if not len(y) or not set(pred) <= {-1, 0, 1}:
        raise ValueError('Invalid schema predictions')
    if variant != 'C_review' and -1 in pred:
        raise ValueError('Binary schema cannot abstain')
    accepted = pred != -1
    n = len(y)
    values = {'coverage': float(accepted.mean()), 'review_rate': float((~accepted).mean()),
              'review_positive_over_all': float(y[~accepted].sum()/n),
              'resolved_correct_over_all': float(((pred == y) & accepted).mean()),
              'known_cost_usd_per_1000': 1000*data['known_cost_usd']/n,
              'unknown_attempts_per_1000': 1000*data['unknown_cost_attempts']/n,
              'accepted_risk': float((pred[accepted] != y[accepted]).mean()) if accepted.any() else None}
    for name, predictions in [('accepted', pred), ('baseline_same_accepted_subset', baseline['pred'])]:
        scores = metric(y[accepted], predictions[accepted]) if accepted.any() else {}
        values.update({name+'_'+k: scores.get(k) for k in ['f1', 'precision', 'recall']})
    return [{'run_id': run_id, 'model': model, 'benchmark': benchmark, 'condition': condition,
             'selected_variant': variant, 'split': 'test', 'n': n, 'accepted_n': int(accepted.sum()),
             'metric': k, 'value': v} for k, v in values.items()]


def export_schema(workspace, module, analyze, baseline_cell, origin, provenance, run_id):
    complete = read(workspace/'artifacts/expanded_schema_complete.json')
    pilot = read(workspace/'artifacts/schema_pilot_status.json')
    if complete.get('status') != 'complete' or pilot.get('complete') is not True:
        raise ValueError('Schema or mandatory pilot incomplete')
    manifest = module.verify()
    expected_pilot = {(m, v, b) for m in module.MODELS for v in module.CONDITIONS
                      for b in manifest['pilot']['families']}
    pilot_keys = [(r['model'], r['condition'], r['benchmark']) for r in pilot['cells']]
    if (len(pilot_keys) != len(expected_pilot) or set(pilot_keys) != expected_pilot
            or any(r['status'] != 'passed' for r in pilot['cells'])):
        raise ValueError('Schema pilot matrix incomplete')
    loaded = {}
    def cell(m, v, b, split, ns='schema'):
        key = (m, v, b, split, ns)
        if key not in loaded:
            loaded[key] = schema_cell(workspace, module, *key, origin, provenance)
        return loaded[key]
    for m, v, b in sorted(expected_pilot):
        cell(m, v, b, 'schema_pilot', 'schema_pilot')
    dev = {(m, v, b): cell(m, v, b, 'prompt') for m in module.MODELS
           for v in module.CONDITIONS for b in module.BENCHMARKS}
    selection, expected_cells = selected_matrix(module, dev, analyze.metric)
    if selection != read(workspace/'configs/expanded_schema_selection.json'):
        raise ValueError('Schema selection differs from full development matrix')
    if (expected_cells != read(workspace/'configs/expanded_schema_test_cells.json')
            or complete.get('test_cells') != len(expected_cells)):
        raise ValueError('Schema test matrix differs')
    rows = []
    for model in module.MODELS:
        for b in [*module.BENCHMARKS, 'billiger-de-50un']:
            baseline = baseline_cell(model, 'B0', b, 'test')[0]
            policies = {'global': selection[model]['global'],
                        'per_benchmark': selection[model]['per_benchmark'][b], 'review': 'C_review'}
            for policy, v in policies.items():
                rows.extend(condition_rows(run_id, model, b, 'schema/'+policy, v,
                                           cell(model, v, b, 'test'), baseline, analyze.metric))
        for b in module.BENCHMARKS:
            for v in ['C_match', 'C_same']:
                a, z = dev[model, v, b], dev[model, v+'_reverse', b]
                if a['cases'] != z['cases'] or not np.array_equal(a['y'], z['y']):
                    raise ValueError('Schema order diagnostic alignment differs')
                values = {'prediction_flip_rate': float((a['pred'] != z['pred']).mean()),
                          'reverse_minus_forward_f1': analyze.metric(z['y'], z['pred'])['f1']
                          - analyze.metric(a['y'], a['pred'])['f1']}
                rows.extend({'run_id': run_id, 'model': model, 'benchmark': b,
                             'condition': 'schema/order/'+v, 'split': 'prompt', 'n': len(a['y']),
                             'metric': k, 'value': value} for k, value in values.items())
    return rows
