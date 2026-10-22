"""Bounded own-run selected tests and final report, never admission or a CLI.

The future wrapper owns scheduler/cost/prerequisite checks and global shared /
replica exclusive locks through development, freeze, tests and report commit.
A failed or interrupted attempt requires a separately reviewed continuation.
"""
import json
import os
from pathlib import Path
import types

from authorized_recovery_200 import make_wrapper
from drained_inference import drain_model_tasks
from fresh_confirmation_selection import RECEIPT, OUTPUTS, durable_write, sha, verify_frozen
from fresh_screen_drained import audit_cell
from operational_journal import inspect_operations
from repeated_runs import shared_account

ATTEMPT = 'artifacts/fresh_confirmation_selected_test'
REPORT_ATTEMPT = 'artifacts/fresh_confirmation_final_report'
MARKER = 'artifacts/expanded_confirmation_complete.json'
REPORT_DIR = 'artifacts/results/expanded_confirmation'
REPORT_NAMES = ('metrics.json', 'macro.json', 'intervals.json', 'unique_test_cell_costs.json',
                'metrics.csv', 'COST_README.txt')


def inventory(target, run_id, confirmation, protocol, cells):
    bound = types.SimpleNamespace(**{**vars(protocol), 'payload': confirmation.payload})
    result = []
    for m, v, b in sorted(set(cells)):
        cases = json.loads((target/'data'/b/'test.json').read_text())
        directory = target/'runs/quality'/m/v/b
        def rows(suffix):
            p = directory/('test'+suffix)
            return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
        failures = rows('.failures.jsonl')
        audit = audit_cell(cases, rows('.jsonl'), failures, protocol,
                           confirmation.payload, m, v, b, run_id)
        operations = inspect_operations(directory/'test.operations.jsonl', cases, failures,
                                        bound, m, v, run_id)
        if operations['unknown_outcome_events']:
            raise ValueError('Unknown test outcomes require terminal review')
        result.append({'model': m, 'variant': v, 'benchmark': b, **audit, 'operations': operations})
    return result


def snapshot(target, cells):
    paths = {target/RECEIPT, *(target/p for p in OUTPUTS)}
    for m, v, b in cells:
        paths.add(target/'data'/b/'test.json')
        for suffix in ('.jsonl', '.failures.jsonl', '.operations.jsonl', '.executions.jsonl'):
            paths.add(target/'runs/quality'/m/v/b/('test'+suffix))
    return {str(p.relative_to(target)): sha(p) for p in sorted(paths)}


def no_report(target):
    out = target/REPORT_DIR
    if ((target/MARKER).exists() or (target/REPORT_ATTEMPT).exists()
            or not out.is_dir()
            or {p.name for p in out.iterdir()} != {'development_metrics.json', 'development_metrics.csv'}):
        raise ValueError('Existing or unexpected report requires review')


def model_completions(target, confirmation):
    events = [json.loads(s) for s in (target/ATTEMPT/'models.jsonl').read_text().splitlines()]
    ends = events[1:-1]
    if (not events or events[0].get('event') != 'start'
            or events[0].get('models') != list(confirmation.MODELS)
            or events[0].get('invocations_per_model') != 1
            or events[-1].get('event') != 'end' or events[-1].get('status') != 'complete'
            or events[-1].get('models_drained') != len(confirmation.MODELS)
            or len(ends) != len(confirmation.MODELS)
            or {e.get('model') for e in ends} != set(confirmation.MODELS)
            or any(e.get('event') != 'model_end' or e.get('status') != 'complete' for e in ends)):
        raise ValueError('All test model completions required')
    return sha(target/ATTEMPT/'models.jsonl')


def run_selected_tests(root, target, run_id, inference, confirmation, expanded, analysis, protocol):
    root, target = Path(root).resolve(), Path(target).resolve()
    candidate = verify_frozen(root, target, run_id, confirmation, expanded, analysis, protocol)
    if inference.PROJECT != target:
        raise ValueError('Wrong inference workspace')
    no_report(target)
    attempt = target/ATTEMPT
    if attempt.exists():
        raise ValueError('Already attempted; new terminal review required')
    cells = candidate['test_cells']
    controls = {(m, 'B0', b) for m, _, b in cells}
    checked_controls = inventory(target, run_id, confirmation, protocol, controls)
    if not all(c['complete'] for c in checked_controls):
        raise ValueError('Complete same-run B0 test controls required')
    before = inventory(target, run_id, confirmation, protocol, cells)
    if any((c['responses'] or c['failure_rows'] or c['operations']['events'])
           and c['variant'] not in ('B0', 'S1') for c in before):
        raise ValueError('Existing expanded test outcomes require terminal review')
    protected = snapshot(target, controls)
    isolated = types.SimpleNamespace(**{**vars(inference), 'payload': confirmation.payload,
                                       'account': shared_account(inference.account, root)})
    _, cell, adapter = make_wrapper(root, 'core').authorized_build(isolated, run_id)
    attempt.mkdir(exist_ok=False)
    # Bind this attempt to the actual durable own-run selection before calls.
    durable_write(attempt/'selection.json', (protocol.canonical({
        'run_id': run_id, 'receipt_sha256': sha(target/RECEIPT),
        'selection_sha256': candidate['selection_sha256'], 'test_cells': cells})+'\n').encode())
    def task(model):
        for m, v, b in cells:
            if m == model:
                cell(m, v, b, 'test', workers=8)
    outcomes = drain_model_tasks(task, list(confirmation.MODELS), attempt/'models.jsonl', protocol.canonical)
    checked = inventory(target, run_id, confirmation, protocol, cells)
    # Complete B0 controls are read-only; the frozen runner may create cache-hit
    # execution records, so only response/failure/operations/input bytes are fixed.
    after = snapshot(target, controls)
    if any(after[p] != h for p, h in protected.items() if not p.endswith('.executions.jsonl')):
        raise ValueError('Selection or B0 controls changed during tests')
    verify_frozen(root, target, run_id, confirmation, expanded, analysis, protocol)
    complete = all(c['complete'] for c in checked) and all(o['status'] == 'complete' for o in outcomes)
    return {'run_id': run_id, 'models': outcomes, 'cells': checked, 'adapter': adapter,
            'test_complete': complete, 'admission_granted': False,
            'report_written': False, 'completion_marker_written': False}


def finish_report(root, target, run_id, confirmation, expanded, analysis, protocol):
    root, target = Path(root).resolve(), Path(target).resolve()
    candidate = verify_frozen(root, target, run_id, confirmation, expanded, analysis, protocol)
    no_report(target)
    model_hash = model_completions(target, confirmation)
    expected_binding = {'run_id': run_id, 'receipt_sha256': sha(target/RECEIPT),
                        'selection_sha256': candidate['selection_sha256'],
                        'test_cells': [list(c) for c in candidate['test_cells']]}
    if json.loads((target/ATTEMPT/'selection.json').read_text()) != expected_binding:
        raise ValueError('Test attempt selection provenance mismatch')
    cells = set(candidate['test_cells']) | {(m, 'B0', b) for m, _, b in candidate['test_cells']}
    checked = inventory(target, run_id, confirmation, protocol, cells)
    if not all(c['complete'] for c in checked):
        raise ValueError('Complete selected tests and B0 controls required')
    before = snapshot(target, cells)
    staging = target/REPORT_ATTEMPT
    staging.mkdir(exist_ok=False)
    out = staging/REPORT_DIR; out.mkdir(parents=True)
    bound = types.SimpleNamespace(**vars(analysis))
    bound.load = types.FunctionType(analysis.load.__code__,
        {**analysis.load.__globals__, 'PROJECT': target, 'payload': confirmation.payload},
        argdefs=analysis.load.__defaults__)
    markers = []
    def capture_marker(path, value):
        if path != staging/MARKER:
            raise ValueError('Unexpected frozen report marker')
        markers.append(value)
    report = types.FunctionType(confirmation.report.__code__,
        {**confirmation.report.__globals__, 'PROJECT': staging, 'analyze': bound,
         'write_frozen': capture_marker}, argdefs=confirmation.report.__defaults__)
    report(candidate['selected'])
    expected = {'status': 'complete', 'selection_sha256': candidate['selection_sha256'],
                'unique_test_cells': len(candidate['test_cells']),
                'selected_regime_rows': len(candidate['selected']), 'full_expanded_study_complete': False}
    if markers != [expected] or {p.name for p in out.iterdir()} != set(REPORT_NAMES):
        raise ValueError('Frozen report output mismatch')
    verify_frozen(root, target, run_id, confirmation, expanded, analysis, protocol)
    if before != snapshot(target, cells) or model_hash != model_completions(target, confirmation):
        raise ValueError('Test sources changed during report')
    for name in REPORT_NAMES:
        durable_write(target/REPORT_DIR/name, (out/name).read_bytes())
    # Persist the report directory before the final frozen completion marker.
    fd = os.open(target/REPORT_DIR, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    durable_write(target/MARKER, (protocol.canonical(expected)+'\n').encode())
    return {str(p.relative_to(target)): sha(p) for p in
            [target/MARKER, *(target/REPORT_DIR/n for n in REPORT_NAMES)]}
