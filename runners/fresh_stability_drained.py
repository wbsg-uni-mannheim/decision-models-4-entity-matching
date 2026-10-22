"""Bounded fresh E3 inference component, without CLI, admission or reports.

The separate stage wrapper must verify successful prerequisites, scheduler,
policy, scopes, costs, budget and joins and retain writer locks through reports.
Repeat namespaces are distinct requests within each fresh study run.
"""
import hashlib
import json
from pathlib import Path
import types

from authorized_recovery_200 import make_wrapper
from drained_inference import drain_model_tasks
from fresh_screen_drained import audit_cell, PAYLOAD_SHA
from operational_journal import inspect_operations
from repeated_runs import replica_path, shared_account

STABILITY_SHA = '1a2d124f5ce7d5304ff15379d622c17e51ac0ced29cbe63e5b8b6bd36d637742'
BENCHMARKS = ['abt-buy', 'wdc', 'dblp-scholar']
INTERACTIONS = {'I3_identity_json': 'X_P_identity_0', 'I3_domain_json': 'X_P_domain_0'}
REPEATS = ['e3_repeat_1', 'e3_repeat_2']


def matrix(stability, controls=False):
    for model in stability.MODELS:
        for benchmark in stability.SCREEN_BENCHMARKS:
            if controls:
                for variant in [*INTERACTIONS.values(), 'B0']:
                    yield 'quality', model, variant, benchmark, 'expanded_screen'
            else:
                for variant in INTERACTIONS:
                    yield 'quality', model, variant, benchmark, 'expanded_screen'
                for namespace in REPEATS:
                    yield namespace, model, 'B0', benchmark, 'expanded_repeat'


def inventory(target, run_id, stability, protocol, controls=False):
    bound = types.SimpleNamespace(**{**vars(protocol), 'payload': stability.payload})
    cells = []
    for namespace, model, variant, benchmark, split in matrix(stability, controls):
        cases = json.loads((target/'data'/benchmark/(split+'.json')).read_text())
        directory = target/'runs'/namespace/model/variant/benchmark
        def rows(suffix):
            path = directory/(split+suffix)
            return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        failures = rows('.failures.jsonl')
        checked = audit_cell(cases, rows('.jsonl'), failures, protocol, stability.payload,
                             model, variant, benchmark, run_id)
        operations = inspect_operations(directory/(split+'.operations.jsonl'), cases, failures,
                                        bound, model, variant, run_id)
        if operations['unknown_outcome_events']:
            raise ValueError('Unknown stability outcomes require separate terminal review')
        cells.append({'namespace': namespace, 'model': model, 'variant': variant,
                      'benchmark': benchmark, 'split': split, **checked, 'operations': operations})
    return cells


def control_hashes(target, stability):
    result = {}
    for ns, m, v, b, split in matrix(stability, controls=True):
        for suffix in ('.jsonl', '.failures.jsonl', '.operations.jsonl', '.executions.jsonl'):
            path = target/'runs'/ns/m/v/b/(split+suffix)
            result[str(path.relative_to(target))] = (hashlib.sha256(path.read_bytes()).hexdigest()
                                                    if path.exists() else None)
    return result


def verify_repeat_separation(target, stability, protocol):
    """Reject copied response records across B0/repeat namespaces; no billing attestation."""
    for b in BENCHMARKS:
        screen = json.loads((target/'data'/b/'expanded_screen.json').read_text())
        repeat = json.loads((target/'data'/b/'expanded_repeat.json').read_text())
        originals = {c['pair_id']: c for c in screen}
        if (len(originals) != len(screen) or len({c['pair_id'] for c in repeat}) != len(repeat)
                or any(originals.get(c['pair_id']) != c for c in repeat)):
            raise ValueError('Repeat input alignment changed')
        for m in stability.MODELS:
            seen = set()
            for ns, split in [('quality', 'expanded_screen'), *[(n, 'expanded_repeat') for n in REPEATS]]:
                path = target/'runs'/ns/m/'B0'/b/(split+'.jsonl')
                for line in path.read_text().splitlines() if path.exists() else []:
                    row = json.loads(line)
                    identity = (row['pair_id'], row['request_hash'], row['completed_at'])
                    if identity in seen:
                        raise ValueError('Copied B0/repeat response across namespaces')
                    seen.add(identity)


def run_stability(root, target, run_id, inference, stability, expanded, protocol, journal_dir):
    root, target, journal_dir = map(lambda p: Path(p).resolve(), (root, target, journal_dir))
    if type(run_id) is not int or run_id not in (2, 3) or target != replica_path(root, run_id):
        raise ValueError('Fresh replica identity required')
    if inference.PROJECT != target or stability.PROJECT != target:
        raise ValueError('Wrong stability workspace')
    for module, digest in ((stability, STABILITY_SHA), (expanded, PAYLOAD_SHA)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen stability source changed')
    if (stability.MODELS != protocol.MODELS or stability.INTERACTIONS != INTERACTIONS
            or stability.SCREEN_BENCHMARKS != BENCHMARKS
            or stability.expanded_payload is not expanded.payload):
        raise ValueError('Frozen stability matrix/payload changed')
    if not journal_dir.is_relative_to(target/'artifacts') or journal_dir.exists():
        raise ValueError('Exclusive replica attempt directory required')
    progress = json.loads((target/'artifacts/progress.json').read_text())
    if progress.get('expanded_E2_E3_screen', {}).get('status') != 'complete':
        raise ValueError('Screen prerequisite incomplete')
    if any((target/p).exists() for p in ('artifacts/results/expanded_stability',
            'artifacts/expanded_stability_complete.json', 'artifacts/expanded_stability_cost_forecast.json')):
        raise ValueError('Existing stability reports require separate terminal review')
    manifest = stability.verify()
    if (manifest['models'] != protocol.MODELS or manifest['new_decisions_before_cache'] != 25200
            or manifest['repeats']['n'] != 100 or manifest['repeats']['additional_runs'] != 2
            or manifest['repeats']['fresh_namespaces'] != REPEATS
            or set(manifest['benchmarks']) != set(BENCHMARKS)):
        raise ValueError('Frozen stability manifest matrix changed')
    for b in BENCHMARKS:
        sample = {'benchmark': b, 'records': {'left': {}, 'right': {}}}
        if manifest['requests'][b] != {m: {v: stability.payload(sample, m, v) for v in INTERACTIONS}
                                      for m in stability.MODELS}:
            raise ValueError('Frozen stability request templates changed')
    controls = inventory(target, run_id, stability, protocol, controls=True)
    if not all(c['complete'] for c in controls):
        raise ValueError('Reused screening controls incomplete')
    protected = control_hashes(target, stability)
    verify_repeat_separation(target, stability, protocol)
    before = inventory(target, run_id, stability, protocol)
    if any(c['failure_rows'] or c['operations']['events'] for c in before):
        raise ValueError('Existing failures require separate terminal review')
    isolated = types.SimpleNamespace(**{**vars(inference), 'payload': stability.payload,
                                       'account': shared_account(inference.account, root)})
    _, cell, adapter = make_wrapper(root, 'core').authorized_build(isolated, run_id)
    journal_dir.mkdir(exist_ok=False)
    def task(model):
        for benchmark in BENCHMARKS:
            for variant in INTERACTIONS:
                cell(model, variant, benchmark, 'expanded_screen', workers=8)
            for namespace in REPEATS:
                cell(model, 'B0', benchmark, 'expanded_repeat', workers=8, namespace=namespace)
    outcomes = drain_model_tasks(task, list(stability.MODELS), journal_dir/'models.jsonl', protocol.canonical)
    checked = inventory(target, run_id, stability, protocol)
    verify_repeat_separation(target, stability, protocol)
    if control_hashes(target, stability) != protected:
        raise ValueError('Reused screening controls changed during stability execution')
    return {'run_id': run_id, 'models': outcomes, 'cells': checked, 'controls': controls,
            'protected_controls_sha256_or_absent': protected, 'adapter': adapter,
            'inference_complete': all(c['complete'] for c in checked) and
                                  all(m['status'] == 'complete' for m in outcomes),
            'admission_granted': False, 'report_written': False, 'completion_marker_written': False,
            'historical_unjournaled_outcomes_resolved': False}
