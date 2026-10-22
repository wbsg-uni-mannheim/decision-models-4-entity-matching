"""Completion-gated offline Peeters export, using frozen inputs and raw parsers."""
from datetime import datetime
import json
import math

import numpy as np

import peeters_control as control
from peeters_run import EXECUTION, MANIFEST, verify_matrix
from repeated_runs import sha


def completion(root, workspace, run_id):
    path = workspace/'artifacts/peeters_stage.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    if len(rows) != 2 or [r['status'] for r in rows] != ['started', 'complete']:
        raise ValueError('Peeters stage incomplete or unsupported recovery history')
    expected = {'run_id': run_id, 'stage': 'peeters', 'target': str(workspace),
                'execution_manifest_sha256': sha(root/EXECUTION), 'budget_cap_usd': 100}
    for row in rows:
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError('Peeters completion identity drift')
        if not all(row.get(k) for k in ['slurm_job_id', 'parallel_policy_sha256', 'admission_sha256']):
            raise ValueError('Peeters admission provenance missing')
    if {k:v for k,v in rows[0].items() if k not in ['utc', 'status']} != {k:v for k,v in rows[1].items() if k not in ['utc', 'status']}:
        raise ValueError('Peeters stage provenance differs')
    times = [datetime.fromisoformat(row['utc']) for row in rows]
    if any(t.tzinfo is None for t in times) or times[0] >= times[1]:
        raise ValueError('Invalid Peeters stage chronology')
    return times, sha(path)


def load_cell(workspace, protocol, model, variant, benchmark, split, namespace,
              run_id, origin, provenance, manifest_sha=None, interval=None):
    cases = json.loads((workspace/'data'/benchmark/(split+'.json')).read_text())
    by_case = {c['pair_id']: c for c in cases}
    if not cases or len(by_case) != len(cases):
        raise ValueError('Empty or duplicate source cases')
    path = workspace/'runs'/namespace/model/variant/benchmark/(split+'.jsonl')
    failure_path = path.with_name(split+'.failures.jsonl')
    raw = path.read_bytes()
    failed_raw = failure_path.read_bytes() if failure_path.exists() else None
    successes = [json.loads(x) for x in raw.splitlines()]
    failures = [json.loads(x) for x in failed_raw.splitlines()] if failed_raw else []
    by_id = {r['pair_id']: r for r in successes}
    if len(by_id) != len(successes) or by_id.keys() != by_case.keys():
        raise ValueError('Incomplete or duplicate response cell')
    origin(successes+failures)
    all_attempts = []
    for rows, is_failure in [(successes, False), (failures, True)]:
        for row in rows:
            c = by_case.get(row['pair_id'])
            if c is None:
                raise ValueError('Failure outside frozen pair set')
            body = control.payload(c, model, variant, protocol) if namespace == 'peeters' else protocol.payload(c, model, variant)
            if (row['request_hash'] != protocol.digest(body) or row['input_hash'] != c['input_hash']
                    or protocol.digest(c['records']) != c['input_hash']
                    or row['model_alias'] != model or row['variant'] != variant or row['benchmark'] != benchmark):
                raise ValueError('Response input/request/condition provenance drift')
            if namespace == 'peeters':
                if (type(row.get('study_run_id')) is not int or row['study_run_id'] != run_id
                        or row.get('peeters_payload_manifest_sha256') != manifest_sha):
                    raise ValueError('Peeters run or payload-manifest identity drift')
                time = datetime.fromisoformat(row['completed_at'])
                if time.tzinfo is None or not interval[0] <= time <= interval[1]:
                    raise ValueError('Response outside admitted stage interval')
            attempts = row['attempts']
            if not attempts or bool(row.get('error')) != is_failure:
                raise ValueError('Invalid success/failure journal placement')
            for a in attempts:
                for k in ['cost', 'seconds']:
                    v = a[k]
                    if k == 'cost' and v is None:
                        continue
                    if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                        raise ValueError('Invalid cost or duration')
            last = attempts[-1]
            if is_failure:
                if not last.get('error'):
                    raise ValueError('Failure journal lacks final attempt error')
            else:
                if last.get('error') or last['status'] != 200 or last['cost'] is None:
                    raise ValueError('Invalid successful final attempt')
                pred, prob = protocol.parse_response(model, last['response'])
                if (row['prediction'] != pred or row['probability_match'] != prob
                        or last['response'].get('usage', {}).get('cost') != last['cost']):
                    raise ValueError('Raw parser or successful usage mismatch')
            all_attempts.extend(attempts)
    ordered = [by_id[c['pair_id']] for c in cases]
    import hashlib
    provenance.append({'model': model, 'variant': variant, 'benchmark': benchmark, 'split': split,
                       'namespace': namespace, 'cache_sha256': hashlib.sha256(raw).hexdigest(),
                       'failure_sha256': hashlib.sha256(failed_raw).hexdigest() if failed_raw is not None else None,
                       'response_count': len(successes), 'failure_rows': len(failures),
                       'request_hashes_verified': True, 'raw_parser_verified': True,
                       'payload_manifest_sha256': manifest_sha})
    return {'cases': cases, 'y': np.array([c['label'] for c in cases]),
            'pred': np.array([r['prediction'] for r in ordered]),
            'known': math.fsum(a['cost'] for a in all_attempts if a['cost'] is not None),
            'unknown': sum(a['cost'] is None for a in all_attempts),
            'seconds': math.fsum(a['seconds'] for a in all_attempts)}


def contrast(a, b, metric):
    if a['cases'] != b['cases'] or not np.array_equal(a['y'], b['y']):
        raise ValueError('Peeters paired inputs or labels differ')
    return {'f1_delta': metric(a['y'], a['pred'])['f1']-metric(b['y'], b['pred'])['f1'],
            'prediction_flip_rate': float((a['pred'] != b['pred']).mean())}


def export_peeters(root, workspace, protocol, metric, normalize, origin, provenance, run_id):
    interval, journal_sha = completion(root, workspace, run_id)
    execution = json.loads((root/EXECUTION).read_text())
    manifest_sha = sha(root/MANIFEST)
    if execution['payload_manifest_sha256'] != manifest_sha:
        raise ValueError('Peeters execution/payload manifest drift')
    for rel, expected in execution['source_sha256'].items():
        if sha(root/rel) != expected or (rel.startswith('src/') and sha(workspace/rel) != expected):
            raise ValueError('Peeters execution source drift')
    matrix = verify_matrix(root, workspace, json.loads((root/MANIFEST).read_text()), protocol)
    loaded, rows = {}, []
    # The complete frozen development matrix is read and checked before test.
    # This checks export ordering, not an independent reconstruction of start times.
    for split in ['prompt', 'route', 'test']:
        for c in [c for c in matrix if c['split'] == split]:
            m, v, b = c['model'], c['arm'], c['benchmark']
            d = load_cell(workspace, protocol, m, v, b, split, 'peeters', run_id,
                          origin, provenance, manifest_sha, interval)
            loaded[m, v, b, split] = d
            rows.extend(normalize(run_id, m, 'peeters/'+v+'/'+split, b, metric(d['y'], d['pred']),
                                  d['known'], d['unknown'], d['seconds'], v))
    test_keys = sorted({(c['model'], c['benchmark']) for c in matrix if c['split'] == 'test'})
    for m, b in test_keys:
        baseline = load_cell(workspace, protocol, m, 'B0', b, 'test', 'quality', run_id, origin, provenance)
        rows.extend(normalize(run_id, m, 'peeters/B0/test', b, metric(baseline['y'], baseline['pred']),
                              baseline['known'], baseline['unknown'], baseline['seconds'], 'B0'))
        pairs = [(v+'_vs_B0', loaded[m, v, b, 'test'], baseline) for v in control.ARMS]
        pairs.append(('values_vs_fields', loaded[m, 'peeters_values', b, 'test'], loaded[m, 'peeters_fields', b, 'test']))
        for name, a, z in pairs:
            rows.extend({'run_id': run_id, 'model': m, 'benchmark': b, 'condition': 'peeters/'+name,
                         'split': 'test', 'n': len(a['y']), 'metric': k, 'value': value,
                         'scope': 'Fixed descriptive paired contrast; no quality selection or historical full replication'}
                        for k, value in contrast(a, z, metric).items())
    return rows, {'stage_journal_sha256': journal_sha, 'execution_manifest_sha256': sha(root/EXECUTION),
                  'payload_manifest_sha256': manifest_sha, 'peeters_cells': len(matrix),
                  'baseline_test_cells': len(test_keys), 'development_loaded_before_test': True,
                  'independent_execution_start_chronology_reconstructed': False}
