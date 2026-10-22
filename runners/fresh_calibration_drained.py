"""Fresh calibration-extra inference only; no admission, fits, reports or CLI.

A separately reviewed wrapper must hold writer locks and prove actual terminal
prerequisites, scheduler/policy/scopes, costs, budget and all dependency joins.
Only missing B0 calibration_extra responses are requested. Reused route caches
are checked but never written; fits and test evaluation belong to the wrapper.
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

CALIBRATION_SHA = '18c9ba45a4f30f806c5b8adc225386a3b9ab47ec2a6947d5a9e555e5de16111d'


def inventory(target, run_id, protocol, manifest, split='calibration_extra'):
    """Raw/input/request/run-id audit, including prior failed attempts and ops."""
    if split not in ('calibration_extra', 'route'):
        raise ValueError('Unexpected calibration source split')
    cells = []
    for model in manifest['models']:
        for benchmark, info in manifest['benchmarks'].items():
            if split == 'calibration_extra' and not info['extra_n']:
                continue
            cases = json.loads((target/'data'/benchmark/(split+'.json')).read_text())
            if split == 'calibration_extra' and len(cases) != info['extra_n']:
                raise ValueError('Calibration extra matrix changed')
            directory = target/'runs/quality'/model/'B0'/benchmark
            def rows(suffix):
                path = directory/(split+suffix)
                return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
            failures = rows('.failures.jsonl')
            checked = audit_rows(cases, rows('.jsonl'), failures, protocol, model, benchmark, run_id)
            operations = inspect_operations(directory/(split+'.operations.jsonl'), cases, failures,
                                            protocol, model, 'B0', run_id)
            if operations['unknown_outcome_events']:
                raise ValueError('Unknown calibration outcomes require separate terminal review')
            cells.append({'model': model, 'benchmark': benchmark, 'split': split,
                          **checked, 'operations': operations})
    return cells


def run_calibration_extra(root, target, run_id, inference, calibration, protocol, journal_dir):
    """One invocation per decision model with the frozen eight cell workers."""
    root, target, journal_dir = Path(root).resolve(), Path(target).resolve(), Path(journal_dir).resolve()
    if type(run_id) is not int or run_id not in (2, 3) or target != replica_path(root, run_id):
        raise ValueError('Fresh replica identity required')
    if inference.PROJECT != target or calibration.PROJECT != target:
        raise ValueError('Wrong calibration workspace')
    if hashlib.sha256(Path(calibration.__file__).read_bytes()).hexdigest() != CALIBRATION_SHA:
        raise ValueError('Frozen calibration source changed')
    decision_models = [m for m in protocol.MODELS if m != 'luna']
    if (calibration.DECISION_MODELS != decision_models
            or calibration.BENCHMARKS != protocol.BENCHMARKS
            or calibration.payload is not protocol.payload):
        raise ValueError('Calibration protocol changed')
    if not journal_dir.is_relative_to(target/'artifacts') or journal_dir.exists():
        raise ValueError('Exclusive replica attempt directory required')
    progress = json.loads((target/'artifacts/progress.json').read_text())
    if progress.get('expanded_E2_E3_screen', {}).get('status') != 'complete':
        raise ValueError('Screen prerequisite incomplete')
    for relative in ('configs/calibration_selection.json', 'artifacts/expanded_calibration_complete.json',
                     'artifacts/results/expanded_calibration', 'artifacts/calibration_cost_forecast.json'):
        if (target/relative).exists():
            raise ValueError('Existing calibration stage requires separate terminal review')
    manifest = calibration.verify()
    if (manifest['models'] != decision_models
            or set(manifest['benchmarks']) != set(protocol.BENCHMARKS)
            or manifest['new_decisions'] != len(manifest['models'])*sum(
                info['extra_n'] for info in manifest['benchmarks'].values())):
        raise ValueError('Calibration manifest matrix changed')
    controls = inventory(target, run_id, protocol, manifest, 'route')
    if not all(c['complete'] for c in controls):
        raise ValueError('Reused route prerequisite incomplete')
    before = inventory(target, run_id, protocol, manifest)
    if any(c['failure_rows'] or c['operations']['events'] for c in before):
        raise ValueError('Existing failures require separate terminal review')
    isolated = types.SimpleNamespace(**{**vars(inference),
                                       'account': shared_account(inference.account, root)})
    _, cell, adapter = make_wrapper(root, 'core').authorized_build(isolated, run_id)
    journal_dir.mkdir(exist_ok=False)
    def task(model):
        for benchmark, info in manifest['benchmarks'].items():
            if info['extra_n']:
                cell(model, 'B0', benchmark, 'calibration_extra', workers=8)
    outcomes = drain_model_tasks(task, list(manifest['models']), journal_dir/'models.jsonl', protocol.canonical)
    checked = inventory(target, run_id, protocol, manifest)
    return {'run_id': run_id, 'models': outcomes, 'cells': checked, 'reused_route_cells': controls,
            'adapter': adapter, 'inference_complete': all(c['complete'] for c in checked) and
            all(m['status'] == 'complete' for m in outcomes), 'admission_granted': False,
            'selection_written': False, 'report_written': False, 'completion_marker_written': False,
            'historical_unjournaled_outcomes_resolved': False}
