"""Fresh supplement inference component, without CLI, admission or reporting.

A future bounded stage wrapper must prove successful terminal core steps, hold
both writer locks, and check current scheduler/policy/scopes/costs/all joins.
This component alone must never be used as admission. Imported frozen modules,
core journals, report files and completion markers are not changed here.
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

SUPPLEMENT_SHA = '4c81624f08f835d430c724ad40b1ab00f9c0244f00e4ec65eced3b2555e06bcc'
BENCHMARK = 'billiger-de-50un'


def inventory(target, run_id, protocol):
    """Read-only input/request/raw-parser and operational-join inspection."""
    cases = json.loads((target/'data'/BENCHMARK/'test.json').read_text())
    cells = []
    for model in protocol.MODELS:
        directory = target/'runs/quality'/model/'B0'/BENCHMARK
        def rows(suffix):
            p = directory/('test'+suffix)
            return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
        failures = rows('.failures.jsonl')
        checked = audit_rows(cases, rows('.jsonl'), failures, protocol, model, BENCHMARK, run_id)
        operations = inspect_operations(directory/'test.operations.jsonl', cases, failures,
                                        protocol, model, 'B0', run_id)
        if operations['unknown_outcome_events']:
            raise ValueError('Unknown supplement outcomes require separate terminal review')
        cells.append({'model': model, **checked, 'operations': operations})
    return cells


def run_supplement(root, target, run_id, inference, supplement, protocol, journal_dir):
    """At most one invocation per model; caller owns actual admission/locks."""
    root, target, journal_dir = Path(root).resolve(), Path(target).resolve(), Path(journal_dir).resolve()
    if type(run_id) is not int or run_id not in (2, 3) or target != replica_path(root, run_id):
        raise ValueError('Fresh replica identity required')
    if inference.PROJECT != target or supplement.PROJECT != target:
        raise ValueError('Wrong supplement workspace')
    if hashlib.sha256(Path(supplement.__file__).read_bytes()).hexdigest() != SUPPLEMENT_SHA:
        raise ValueError('Frozen supplement source changed')
    if supplement.BENCHMARK != BENCHMARK or supplement.MODELS != protocol.MODELS:
        raise ValueError('Supplement protocol changed')
    if not journal_dir.is_relative_to(target/'artifacts') or journal_dir.exists():
        raise ValueError('Exclusive replica attempt directory required')
    progress = json.loads((target/'artifacts/progress.json').read_text())
    if progress.get('AP3_AP4_analysis', {}).get('status') != 'complete_automated':
        raise ValueError('Core analysis prerequisite incomplete')
    if 'supplement_billiger50' in progress:
        raise ValueError('Existing supplement stage requires separate terminal review')
    if (target/'artifacts/results/billiger50').exists():
        raise ValueError('Existing supplement reports require separate review')
    supplement.verify()
    before = inventory(target, run_id, protocol)
    if any(c['failure_rows'] or c['operations']['events'] for c in before):
        raise ValueError('Existing failures require separate terminal review')
    isolated = types.SimpleNamespace(**{**vars(inference),
                                       'account': shared_account(inference.account, root)})
    # Reuse the already tested authority and private atomic USD200 account.
    # Constructing this binding does not call its core wrapper or grant admission.
    _, cell, adapter = make_wrapper(root, 'core').authorized_build(isolated, run_id)
    journal_dir.mkdir(exist_ok=False)
    def task(model):
        cell(model, 'B0', BENCHMARK, 'test', workers=16)
    outcomes = drain_model_tasks(task, list(protocol.MODELS), journal_dir/'models.jsonl', protocol.canonical)
    checked = inventory(target, run_id, protocol)
    return {'run_id': run_id, 'models': outcomes, 'cells': checked, 'adapter': adapter,
            'inference_complete': all(c['complete'] for c in checked) and
                                  all(m['status'] == 'complete' for m in outcomes),
            'admission_granted': False, 'report_written': False, 'completion_marker_written': False,
            'historical_unjournaled_outcomes_resolved': False}
