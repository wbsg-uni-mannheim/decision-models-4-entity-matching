"""Schema-specific execution component for a separately admitted Run1 recovery.

No CLI, admission, pilot repetition, report or completion marker. The future
wrapper must hold the schema writer lock and check current scheduler, policy,
scope, diagnosis, caches, sources, costs, budget and all dependency joins.
Frozen modules remain untouched. Historical unjournaled outcomes remain unknown.
"""
import hashlib
import json
from pathlib import Path
import threading
import types

from diagnose_schema_terminal import audit_schema_rows
from drained_inference import build, drain_model_tasks
from operational_journal import inspect_operations
from schema_run_export import selected_matrix

SCHEMA_SHA = 'aaf916eb209e125b340fe30d08456b4d6fa58de756708414a0c1224538e737b1'


def source_check(schema):
    if hashlib.sha256(Path(schema.__file__).read_bytes()).hexdigest() != SCHEMA_SHA:
        raise ValueError('Frozen schema source changed')


def build_schema(schema, inference):
    """Bind exact wire bytes and a private parser context inside each worker."""
    source_check(schema)
    context = threading.local()

    def parse(alias, raw):
        return schema.parse(alias, raw, context.variant)

    isolated = types.SimpleNamespace(**{**vars(inference), 'payload': schema.payload,
        'canonical': schema.wire, 'digest': schema.wire_hash, 'parse_response': parse})
    request, cell, provenance = build(isolated, 1)

    def schema_request(case, alias, variant, credential):
        if variant not in schema.CONDITIONS:
            raise ValueError('Not a frozen schema condition')
        context.variant = variant
        try:
            return request(case, alias, variant, credential)
        finally:
            del context.variant

    # This is the private generated function namespace, never run.__dict__.
    cell.__globals__['request'] = schema_request
    return schema_request, cell, {**provenance, 'schema_sha256': SCHEMA_SHA,
        'wire_binding': 'Exact frozen insertion-order serialization and SHA256',
        'parser_binding': 'Private thread-local variant set inside request worker'}


def selected_tests(root, schema, analysis):
    """Audit complete pilot/development and reconstruct selection, read-only."""
    if schema.PROJECT != root:
        raise ValueError('Wrong schema workspace')
    source_check(schema)
    manifest = schema.verify()
    pilot_file = root/'artifacts/schema_pilot_status.json'
    pilot = json.loads(pilot_file.read_text())
    expected = {(m, v, b) for m in schema.MODELS for v in schema.CONDITIONS
                for b in manifest['pilot']['families']}
    observed = [(r['model'], r['condition'], r['benchmark']) for r in pilot['cells']]
    if (pilot.get('complete') is not True or len(observed) != len(expected)
            or set(observed) != expected or any(r['status'] != 'passed' for r in pilot['cells'])):
        raise ValueError('Mandatory pilot incomplete')
    loader = types.FunctionType(schema.load.__code__,
        {**schema.load.__globals__, 'PROJECT': root}, argdefs=schema.load.__defaults__)

    def load(m, v, b, split, namespace='schema'):
        cases = json.loads((root/'data'/b/(split+'.json')).read_text())
        directory = root/'runs'/namespace/m/v/b
        successes = [json.loads(x) for x in (directory/(split+'.jsonl')).read_text().splitlines()]
        failure = directory/(split+'.failures.jsonl')
        failures = [json.loads(x) for x in failure.read_text().splitlines()] if failure.exists() else []
        audit = audit_schema_rows(cases, successes, failures, schema, m, v, b)
        if not audit['complete']:
            raise ValueError('Incomplete schema source cell')
        bound = types.SimpleNamespace(payload=schema.payload, digest=schema.wire_hash)
        operations = inspect_operations(directory/(split+'.operations.jsonl'), cases, failures, bound, m, v, 1)
        if operations['unknown_outcome_events']:
            raise ValueError('Unresolved schema source operations')
        return loader(m, v, b, split, namespace)

    for m, v, b in sorted(expected):
        load(m, v, b, 'schema_pilot', 'schema_pilot')
    development = {(m, v, b): load(m, v, b, 'prompt') for m in schema.MODELS
                   for v in schema.CONDITIONS for b in schema.BENCHMARKS}
    selected, cells = selected_matrix(schema, development, analysis.metric)
    selection_file = root/'configs/expanded_schema_selection.json'
    cells_file = root/'configs/expanded_schema_test_cells.json'
    if (selected != json.loads(selection_file.read_text())
            or cells != json.loads(cells_file.read_text())):
        raise ValueError('Frozen schema selection or test matrix mismatch')
    return cells, {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (pilot_file, selection_file, cells_file)}


def run_selected_tests(root, schema, analysis, inference, journal_dir):
    """One test invocation per model; only a separately admitted caller may run."""
    if inference.PROJECT != root:
        raise ValueError('Wrong inference workspace')
    if (root/'artifacts/expanded_schema_complete.json').exists():
        raise ValueError('Schema already complete')
    if journal_dir.exists():
        raise ValueError('Already attempted; new terminal review required')
    cells, hashes = selected_tests(root, schema, analysis)
    _, cell, adapter = build_schema(schema, inference)
    journal_dir.mkdir(exist_ok=False)

    def task(model):
        for alias, variant, benchmark in cells:
            if alias == model:
                cell(alias, variant, benchmark, 'test', workers=8, namespace='schema')

    outcomes = drain_model_tasks(task, list(schema.MODELS), journal_dir/'models.jsonl', schema.wire)
    bound = types.SimpleNamespace(payload=schema.payload, digest=schema.wire_hash)
    operations = []
    for model, variant, benchmark in cells:
        directory = root/'runs/schema'/model/variant/benchmark
        failure = directory/'test.failures.jsonl'
        rows = [json.loads(x) for x in failure.read_text().splitlines()] if failure.exists() else []
        cases = json.loads((root/'data'/benchmark/'test.json').read_text())
        operations.append({'model': model, 'variant': variant, 'benchmark': benchmark,
            **inspect_operations(directory/'test.operations.jsonl', cases, rows, bound, model, variant, 1)})
    if any(hashlib.sha256((root/p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise ValueError('Schema selection or pilot status changed during execution')
    return {'models': outcomes, 'operations': operations, 'adapter': adapter,
        'selection_file_sha256': hashes, 'test_cells': len(cells),
        'completion_marker_written': False, 'recovery_admitted': False,
        'historical_unjournaled_outcomes_resolved': False}
