"""Fresh fixed-rule/manual-control inference only, with no CLI or admission.

A future wrapper must prove completed prerequisites and their entire histories,
check current scheduler/scopes/sources/cache/cost/budget/joins, hold both locks,
and privately bind grouped writes and credits before any real execution.
"""
import json
from pathlib import Path
import types

from authorized_recovery_200 import make_wrapper
from diagnose_fresh_baseline import audit_rows
from drained_inference import drain_model_tasks
from operational_journal import inspect_operations
from repeated_runs import replica_path, shared_account
from rules_drained import prepare, rows, sha, unchanged

SUFFIXES = ('.jsonl', '.failures.jsonl', '.operations.jsonl', '.executions.jsonl')


def audit(target, run_id, frozen, protocol, cells, require_complete=False):
    results = []
    for c in cells:
        m, v, b, split, ns = (c[k] for k in ('model', 'variant', 'benchmark', 'split', 'namespace'))
        cases = json.loads((target/'data'/b/(split+'.json')).read_text())
        directory = target/'runs'/ns/m/v/b
        successes, failures = (rows(directory/(split+s)) for s in ('.jsonl', '.failures.jsonl'))
        for r in successes+failures:
            if r.get('variant') != v or type(r.get('study_run_id')) is not int or r['study_run_id'] != run_id:
                raise ValueError('Foreign fresh rules/control variant or run')
        # B0 audit implementation reused only after exact variant/run validation;
        # request is the actual fixed-rule/manual payload, no persisted row edits.
        bound = types.SimpleNamespace(digest=protocol.digest, parse_response=protocol.parse_response,
            payload=lambda case, alias, _: frozen.payload(case, alias, v))
        checked = audit_rows(cases, [{**r, 'variant': 'B0'} for r in successes],
                             [{**r, 'variant': 'B0'} for r in failures], bound, m, b, run_id)
        ops = inspect_operations(directory/(split+'.operations.jsonl'), cases, failures,
            types.SimpleNamespace(digest=protocol.digest, payload=frozen.payload), m, v, run_id)
        if ops['unknown_outcome_events']:
            raise ValueError('Unknown rules/control outcome requires separate terminal review')
        if require_complete and not checked['complete']:
            raise ValueError('Incomplete fresh rules/control phase')
        results.append({**c, **checked, 'operations': ops})
    return results


def hashes(target, cells):
    return {str(p.relative_to(target)): sha(p) if p.exists() else None
            for c in cells for s in SUFFIXES
            for p in [target/'runs'/c['namespace']/c['model']/c['variant']/c['benchmark']/(c['split']+s)]}


def untouched(target, protocol):
    history = target/'artifacts/stage_execution.jsonl'
    if history.exists() and any(r.get('stage') == 'rules' for r in rows(history)):
        raise ValueError('Existing rules stage requires separate terminal review')
    for rel in ('artifacts/expanded_rules_complete.json', 'artifacts/results/expanded_rules',
                'artifacts/rules_cost_forecast.json'):
        if (target/rel).exists():
            raise ValueError('Existing rule output requires separate review')
    # Manual-control journals from the own-run confirmation are legitimate;
    # their full raw audit below is mandatory, not a namespace-only cache credit.
    if list((target/'runs/rules').rglob('*.jsonl')):
        raise ValueError('First rules attempt requires absent rule journals')


def prerequisites(target):
    progress = json.loads((target/'artifacts/progress.json').read_text())
    for key in ('AP3_AP4_analysis', 'supplement_billiger50'):
        if progress.get(key, {}).get('status') != 'complete_automated':
            raise ValueError('Core/supplement prerequisite incomplete')
    for stage in ('confirmation', 'schema'):
        p = target/f'artifacts/expanded_{stage}_complete.json'
        if not p.exists() or json.loads(p.read_text()).get('status') != 'complete':
            raise ValueError('Shared-cache prerequisite incomplete: '+stage)


def run_rules(root, target, run_id, inference, rules, protocol, journal_dir):
    root, target, journal_dir = (Path(p).resolve() for p in (root, target, journal_dir))
    if type(run_id) is not int or run_id not in (2, 3) or target != replica_path(root, run_id):
        raise ValueError('Fresh replica identity required')
    if inference.PROJECT != target or rules.PROJECT != target:
        raise ValueError('Wrong rules workspace')
    if not journal_dir.is_relative_to(target/'artifacts') or journal_dir.exists():
        raise ValueError('Exclusive fresh attempt directory required')
    prerequisites(target)
    untouched(target, protocol)
    frozen, spec, cells, protected = prepare(target, rules, protocol)
    controls = [dict(c, variant='B0', namespace='quality') for c in cells if c['variant'] == 'R_rules']
    control_hashes = hashes(target, controls)
    before = audit(target, run_id, frozen, protocol, cells)
    audit(target, run_id, frozen, protocol, controls, require_complete=True)
    if hashes(target, controls) != control_hashes:
        raise ValueError('Own B0 controls changed during initial validation')
    unchanged(target, protected)
    isolated = types.SimpleNamespace(**{**vars(inference), 'payload': frozen.payload,
        'account': shared_account(inference.account, root)})
    _, cell, adapter = make_wrapper(root, 'core').authorized_build(isolated, run_id)
    journal_dir.mkdir(exist_ok=False)
    result = {'run_id': run_id, 'phases': [], 'before': before, 'adapter': adapter,
              'protected_sha256': protected, 'protected_controls_sha256_or_absent': control_hashes,
              'admission_granted': False, 'report_written': False, 'completion_marker_written': False}
    for split in ('prompt', 'test'):
        def task(model):
            for b in [*protocol.BENCHMARKS, *(['billiger-de-50un'] if split == 'test' else [])]:
                for v, ns in frozen.CONDITIONS.items():
                    cell(model, v, b, split, workers=8, namespace=ns)
        outcomes = drain_model_tasks(task, list(protocol.MODELS), journal_dir/(split+'_models.jsonl'), protocol.canonical)
        checked = audit(target, run_id, frozen, protocol, [c for c in cells if c['split'] == split])
        unchanged(target, protected)
        if hashes(target, controls) != control_hashes:
            raise ValueError('Own B0 controls changed during rules execution')
        complete = all(c['complete'] for c in checked) and all(m['status'] == 'complete' for m in outcomes)
        result['phases'].append({'split': split, 'models': outcomes, 'cells': checked, 'complete': complete})
        if not complete:
            result['inference_complete'] = False
            return result
    # Complete development precedes every test invocation. Recheck both phases
    # before returning; a future wrapper must repeat this under its report locks.
    audit(target, run_id, frozen, protocol, cells, require_complete=True)
    audit(target, run_id, frozen, protocol, controls, require_complete=True)
    unchanged(target, protected)
    result['inference_complete'] = True
    return result
