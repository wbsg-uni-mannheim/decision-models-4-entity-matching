"""Fresh E6 inference only: no CLI, admission, scheduler, reports or markers.

The future bounded wrapper owns full terminal core/history checks, both locks,
current scope/source/cost/budget/join gates, and grouped credit reconciliation.
This first-attempt component never resumes existing variant journals.
"""
import hashlib
import json
from pathlib import Path
import types

from authorized_recovery_200 import make_wrapper
from diagnose_fresh_baseline import audit_rows
from drained_inference import drain_model_tasks
from operational_journal import inspect_operations
from repeated_runs import replica_path, shared_account

VARIANTS_SHA = '05c931c68ed829d40e9c84ff6ea1cd09d61b22aecbfa1c351248451b4b233b8e'
VARIANTS = {'billiger-en-en-100un': 'billiger-de', 'billiger-de-en-100un': 'billiger-de',
            'billiger-en-de-100un': 'billiger-de', 'wdc-000un': 'wdc', 'wdc-050un': 'wdc'}
CONTROLS = ('billiger-de', 'wdc')
SUFFIXES = ('.jsonl', '.failures.jsonl', '.operations.jsonl', '.executions.jsonl')


def inventory(target, run_id, protocol, controls=False):
    cells = []
    for model in protocol.MODELS:
        for benchmark in CONTROLS if controls else VARIANTS:
            cases = json.loads((target/'data'/benchmark/'test.json').read_text())
            directory = target/'runs/quality'/model/'B0'/benchmark
            def rows(suffix):
                p = directory/('test'+suffix)
                return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
            failures = rows('.failures.jsonl')
            checked = audit_rows(cases, rows('.jsonl'), failures, protocol, model, benchmark, run_id)
            operations = inspect_operations(directory/'test.operations.jsonl', cases, failures,
                                            protocol, model, 'B0', run_id)
            if operations['unknown_outcome_events']:
                raise ValueError('Unknown variant/control outcomes require separate terminal review')
            cells.append({'model': model, 'variant': 'B0', 'benchmark': benchmark, 'split': 'test',
                          **checked, 'operations': operations})
    return cells


def control_hashes(target, protocol):
    result = {}
    for m in protocol.MODELS:
        for b in CONTROLS:
            for suffix in SUFFIXES:
                p = target/'runs/quality'/m/'B0'/b/('test'+suffix)
                result[str(p.relative_to(target))] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    return result


def untouched(target, protocol):
    history = target/'artifacts/stage_execution.jsonl'
    if history.exists() and any(json.loads(s).get('stage') == 'variants' for s in history.read_text().splitlines()):
        raise ValueError('Existing variant stage requires separate terminal review')
    for rel in ('artifacts/expanded_variants_complete.json', 'artifacts/results/expanded_variants',
                'artifacts/expanded_variants_cost_forecast.json'):
        if (target/rel).exists():
            raise ValueError('Existing variant output requires separate review')
    for m in protocol.MODELS:
        for b in VARIANTS:
            if list((target/'runs/quality'/m/'B0'/b).glob('*.jsonl')):
                raise ValueError('First variant attempt requires absent journals')


def run_variants(root, target, run_id, inference, variants, protocol, journal_dir):
    root, target, journal_dir = (Path(p).resolve() for p in (root, target, journal_dir))
    if type(run_id) is not int or run_id not in (2, 3) or target != replica_path(root, run_id):
        raise ValueError('Fresh replica identity required')
    if inference.PROJECT != target or variants.PROJECT != target:
        raise ValueError('Wrong variants workspace')
    if hashlib.sha256(Path(variants.__file__).read_bytes()).hexdigest() != VARIANTS_SHA:
        raise ValueError('Frozen variants source changed')
    if list(variants.VARIANTS.items()) != list(VARIANTS.items()) or variants.MODELS != protocol.MODELS:
        raise ValueError('Frozen variants matrix changed')
    if not journal_dir.is_relative_to(target/'artifacts') or journal_dir.exists():
        raise ValueError('Exclusive replica attempt directory required')
    progress = json.loads((target/'artifacts/progress.json').read_text())
    if progress.get('AP3_AP4_analysis', {}).get('status') != 'complete_automated':
        raise ValueError('Core analysis prerequisite incomplete')
    untouched(target, protocol)
    manifest = variants.verify()
    if (manifest['models'] != protocol.MODELS or manifest['new_decisions_before_cache'] != 156387
            or set(manifest['conditions']) != set(VARIANTS)):
        raise ValueError('Frozen variant manifest matrix changed')
    for b, family in VARIANTS.items():
        sample = {'benchmark': b, 'records': {'left': {}, 'right': {}}}
        info = manifest['conditions'][b]
        if info['family'] != family or info['request_templates'] != {m: protocol.payload(sample, m, 'B0') for m in protocol.MODELS}:
            raise ValueError('Frozen variant request templates changed')
    protected = control_hashes(target, protocol)
    controls = inventory(target, run_id, protocol, controls=True)
    if not all(c['complete'] for c in controls):
        raise ValueError('Own core variant controls incomplete')
    if control_hashes(target, protocol) != protected:
        raise ValueError('Controls changed during validation')
    # Private authorized USD200 per-call binding; the future wrapper must
    # explicitly adapt and test grouped/leased accounting before a real start.
    isolated = types.SimpleNamespace(**{**vars(inference), 'account': shared_account(inference.account, root)})
    _, cell, adapter = make_wrapper(root, 'core').authorized_build(isolated, run_id)
    journal_dir.mkdir(exist_ok=False)
    def task(model):
        for benchmark in VARIANTS:
            cell(model, 'B0', benchmark, 'test', workers=8)
    outcomes = drain_model_tasks(task, list(protocol.MODELS), journal_dir/'models.jsonl', protocol.canonical)
    checked = inventory(target, run_id, protocol)
    if control_hashes(target, protocol) != protected:
        raise ValueError('Own core controls changed during variants execution')
    return {'run_id': run_id, 'models': outcomes, 'cells': checked, 'controls': controls,
            'protected_controls_sha256_or_absent': protected, 'adapter': adapter,
            'inference_complete': all(c['complete'] for c in checked) and all(m['status'] == 'complete' for m in outcomes),
            'admission_granted': False, 'report_written': False, 'completion_marker_written': False,
            'historical_unjournaled_outcomes_resolved': False}
