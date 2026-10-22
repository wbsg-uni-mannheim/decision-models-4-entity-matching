"""Own-run selected schema tests and final report; no CLI or admission.

The future wrapper must hold global shared / replica exclusive locks and
check current scheduler, prerequisites, scopes, costs, budget and all joins.
Only first attempts are supported. Interrupted writes require separate review.
"""
import json
from pathlib import Path
import types

from drained_inference import drain_model_tasks
import fresh_schema_development_drained as development
import fresh_schema_selection as freeze
from fresh_screen_drained import audit_cell as audit_baseline
from operational_journal import inspect_operations

ATTEMPT = 'artifacts/fresh_schema_selected_test'
REPORT_ATTEMPT = 'artifacts/fresh_schema_final_report'
MARKER = 'artifacts/expanded_schema_complete.json'
REPORT_DIR = 'artifacts/results/expanded_schema'
REPORT_NAMES = ('metrics.json', 'development.json', 'order_effects_development.json',
                'macro.json', 'intervals.json', 'costs.json', 'pilot.json')


def paths(target, cells):
    result = set()
    for m, v, b in cells:
        result.add(target/'data'/b/'test.json')
        namespace = 'quality' if v == 'B0' else 'schema'
        result.update(target/'runs'/namespace/m/v/b/('test'+s) for s in freeze.SUFFIXES)
    return {freeze.own_path(target, p) for p in result}


def snapshot(target, cells):
    return {str(p.relative_to(target)): freeze.sha(p) for p in sorted(paths(target, cells))}


def inventory(target, run_id, schema, cells):
    result = []
    protocol = schema.protocol
    for m, v, b in sorted(set(map(tuple, cells))):
        paths(target, [(m, v, b)])  # Reject foreign/symlinked input and journal paths.
        cases = json.loads((target/'data'/b/'test.json').read_text())
        baseline = v == 'B0'
        directory = target/'runs'/('quality' if baseline else 'schema')/m/v/b
        def rows(suffix):
            p = directory/('test'+suffix)
            return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
        failures = rows('.failures.jsonl')
        if baseline:
            audit = audit_baseline(cases, rows('.jsonl'), failures, protocol,
                                   protocol.payload, m, v, b, run_id)
            bound = protocol
        else:
            audit = development.audit_cell(cases, rows('.jsonl'), failures, schema, m, v, b, run_id)
            bound = types.SimpleNamespace(payload=schema.payload, digest=schema.wire_hash)
        operations = inspect_operations(directory/'test.operations.jsonl', cases, failures,
                                        bound, m, v, run_id)
        if operations['unknown_outcome_events']:
            raise ValueError('Unknown test outcomes require terminal review')
        result.append(dict(model=m, variant=v, benchmark=b, **audit, operations=operations))
    return result


def no_report(target):
    if any((target/p).exists() for p in (MARKER, REPORT_DIR, REPORT_ATTEMPT)):
        raise ValueError('Existing report or partial report requires separate review')


def binding(target, run_id, candidate, controls):
    return dict(run_id=run_id, receipt_sha256=freeze.sha(target/freeze.RECEIPT),
                selection_sha256={p:freeze.sha(target/p) for p in freeze.OUTPUTS},
                test_cells=[list(c) for c in candidate['test_cells']],
                baseline_sha256=snapshot(target, controls), component_sha256=freeze.sha(Path(__file__)))


def model_completions(target, schema):
    p = freeze.own_path(target, target/ATTEMPT/'models.jsonl')
    events = [json.loads(s) for s in p.read_text().splitlines()]
    ends = events[1:-1]
    if (not events or events[0].get('event') != 'start'
            or events[0].get('models') != list(schema.MODELS)
            or events[0].get('invocations_per_model') != 1
            or events[-1].get('event') != 'end' or events[-1].get('status') != 'complete'
            or events[-1].get('models_drained') != len(schema.MODELS)
            or len(ends) != len(schema.MODELS)
            or {e.get('model') for e in ends} != set(schema.MODELS)
            or any(e.get('event') != 'model_end' or e.get('status') != 'complete' for e in ends)):
        raise ValueError('All test model completions required')
    return freeze.sha(p)


def run_selected_tests(root, target, run_id, inference, schema, analysis):
    root, target = Path(root).resolve(), Path(target).resolve()
    candidate = freeze.verify_frozen(root, target, run_id, schema, analysis)
    if inference.PROJECT != target:
        raise ValueError('Wrong inference workspace')
    no_report(target)
    attempt = freeze.own_path(target, target/ATTEMPT)
    if attempt.exists():
        raise ValueError('Already attempted; terminal review required')
    cells = candidate['test_cells']
    controls = {(m, 'B0', b) for m, _, b in cells}
    if not all(c['complete'] for c in inventory(target, run_id, schema, controls)):
        raise ValueError('Complete same-run B0 test controls required')
    if any(p.exists() for p in paths(target, cells) if p.suffix == '.jsonl'):
        raise ValueError('Existing schema test journals require terminal review')
    provenance = binding(target, run_id, candidate, controls)
    _, cell, adapter = development.build_schema(root, schema, inference, run_id)
    attempt.mkdir(exist_ok=False)
    freeze.durable_write(attempt/'selection.json', (schema.protocol.canonical(provenance)+'\n').encode())
    def task(model):
        for m, v, b in cells:
            if m == model:
                cell(m, v, b, 'test', workers=8, namespace='schema')
    outcomes = drain_model_tasks(task, list(schema.MODELS), attempt/'models.jsonl', schema.protocol.canonical)
    checked = inventory(target, run_id, schema, cells)
    freeze.verify_frozen(root, target, run_id, schema, analysis)
    if provenance != binding(target, run_id, candidate, controls):
        raise ValueError('Selection or baseline controls changed during tests')
    return dict(run_id=run_id, models=outcomes, cells=checked, adapter=adapter,
                test_complete=all(c['complete'] for c in checked) and all(o['status']=='complete' for o in outcomes),
                admission_granted=False, report_written=False, completion_marker_written=False)


def finish_report(root, target, run_id, schema, analysis):
    root, target = Path(root).resolve(), Path(target).resolve()
    candidate = freeze.verify_frozen(root, target, run_id, schema, analysis)
    no_report(target)
    model_hash = model_completions(target, schema)
    cells = set(map(tuple, candidate['test_cells']))
    controls = {(m, 'B0', b) for m, _, b in cells}
    provenance = binding(target, run_id, candidate, controls)
    provenance_path = freeze.own_path(target, target/ATTEMPT/'selection.json')
    if json.loads(provenance_path.read_text()) != provenance:
        raise ValueError('Test attempt selection provenance mismatch')
    if not all(c['complete'] for c in inventory(target, run_id, schema, cells|controls)):
        raise ValueError('Complete selected tests and B0 controls required')
    before, binding_hash = snapshot(target, cells|controls), freeze.sha(provenance_path)
    staging = freeze.own_path(target, target/REPORT_ATTEMPT)
    staging.mkdir(exist_ok=False)
    for p in (*freeze.OUTPUTS, 'artifacts/schema_pilot_status.json'):
        freeze.durable_write(staging/p, (target/p).read_bytes())
    loader = types.FunctionType(schema.load.__code__, {**schema.load.__globals__, 'PROJECT':target},
                                argdefs=schema.load.__defaults__)
    baseline = types.FunctionType(analysis.load.__code__, {**analysis.load.__globals__, 'PROJECT':target},
                                  argdefs=analysis.load.__defaults__)
    bound = types.SimpleNamespace(**{**vars(analysis), 'load':baseline})
    markers = []
    def capture_marker(path, value):
        if path != staging/MARKER:
            raise ValueError('Unexpected frozen report marker')
        markers.append(value)
    report = types.FunctionType(schema.report.__code__,
        {**schema.report.__globals__, 'PROJECT':staging, 'load':loader, 'analyze':bound,
         'write_frozen':capture_marker}, argdefs=schema.report.__defaults__)
    report()
    expected = dict(status='complete', test_cells=len(cells), full_expanded_study_complete=False)
    out = staging/REPORT_DIR
    if markers != [expected] or {p.name for p in out.iterdir()} != set(REPORT_NAMES):
        raise ValueError('Frozen report output mismatch')
    freeze.verify_frozen(root, target, run_id, schema, analysis)
    if (before != snapshot(target, cells|controls) or model_hash != model_completions(target, schema)
            or binding_hash != freeze.sha(provenance_path)):
        raise ValueError('Test sources changed during report')
    for name in REPORT_NAMES:
        freeze.durable_write(target/REPORT_DIR/name, (out/name).read_bytes())
    freeze.sync_directory(target/REPORT_DIR)
    freeze.durable_write(target/MARKER, (schema.protocol.canonical(expected)+'\n').encode())
    return {str(p.relative_to(target)):freeze.sha(p) for p in
            [target/MARKER, *(target/REPORT_DIR/n for n in REPORT_NAMES)]}
