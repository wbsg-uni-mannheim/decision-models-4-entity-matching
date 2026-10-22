"""Fresh schema pilot/development component, not an admission or launcher.

Preserves the frozen serial pilot order (two request workers) and development
model/condition/family order (seven model tasks, eight request workers). Stops
after the first failed pilot cell, draining that cell's submitted requests.
Development selection is only reconstructed in memory after all 280 cells.
A later wrapper must hold locks and check actual prerequisites, all scheduler
joins, scope, sources, caches, concrete costs and budget before this entry.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import types

from authorized_recovery_200 import make_wrapper
from diagnose_fresh_baseline import audit_rows
from drained_inference import drain_model_tasks, durable_append
from operational_journal import inspect_operations
from repeated_runs import replica_path, shared_account
from schema_drained import source_check, SCHEMA_SHA
from schema_run_export import selected_matrix

PILOT_FAMILIES = ['abt-buy', 'dblp-scholar']
BINARY = ['C_match', 'C_match_reverse', 'C_same', 'C_same_reverse']


def matrix(schema, pilot=False):
    families = PILOT_FAMILIES if pilot else schema.BENCHMARKS
    return [(m, v, b) for m in schema.MODELS for v in schema.CONDITIONS for b in families]


def audit_cell(cases, successes, failures, schema, model, variant, benchmark, run_id):
    if type(run_id) is not int or run_id not in (2, 3):
        raise ValueError('Fresh run identity required')
    if any(r['variant'] != variant for r in successes+failures):
        raise ValueError('Foreign schema variant')
    adapter = types.SimpleNamespace(
        digest=lambda value: hashlib.sha256(value.encode()).hexdigest()
        if isinstance(value, str) else schema.protocol.digest(value),
        payload=lambda case, alias, _: schema.wire(schema.payload(case, alias, variant)),
        parse_response=lambda alias, raw: schema.parse(alias, raw, variant))
    # Only the generic audit's variant argument is adapted, never persisted data.
    # Run IDs stay intact and are strictly required by audit_rows.
    groups = [[{**r, 'variant': 'B0'} for r in group] for group in (successes, failures)]
    return audit_rows(cases, *groups, adapter, model, benchmark, run_id)


def inventory(target, run_id, schema, pilot=False):
    namespace, split = ('schema_pilot', 'schema_pilot') if pilot else ('schema', 'prompt')
    bound = types.SimpleNamespace(payload=schema.payload, digest=schema.wire_hash)
    cases = {b: json.loads((target/'data'/b/(split+'.json')).read_text())
             for b in (PILOT_FAMILIES if pilot else schema.BENCHMARKS)}
    result, expected = [], set()
    for model, variant, benchmark in matrix(schema, pilot):
        directory = target/'runs'/namespace/model/variant/benchmark
        def rows(suffix):
            p = directory/(split+suffix)
            return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
        for suffix in ('.jsonl', '.failures.jsonl', '.operations.jsonl', '.executions.jsonl'):
            expected.add(directory/(split+suffix))
        failures = rows('.failures.jsonl')
        checked = audit_cell(cases[benchmark], rows('.jsonl'), failures, schema,
                             model, variant, benchmark, run_id)
        operations = inspect_operations(directory/(split+'.operations.jsonl'), cases[benchmark],
                                        failures, bound, model, variant, run_id)
        if operations['unknown_outcome_events']:
            raise ValueError('Unknown schema outcomes require terminal review')
        result.append(dict(model=model, variant=variant, benchmark=benchmark,
                           namespace=namespace, split=split, **checked, operations=operations))
    if set((target/'runs'/namespace).rglob('*.jsonl'))-expected:
        raise ValueError('Unexpected schema journals require terminal review')
    return result


def build_schema(root, schema, inference, run_id):
    source_check(schema)
    context = threading.local()
    def parse(alias, raw):
        return schema.parse(alias, raw, context.variant)
    isolated = types.SimpleNamespace(**{**vars(inference), 'payload': schema.payload,
        'canonical': schema.wire, 'digest': schema.wire_hash, 'parse_response': parse,
        'account': shared_account(inference.account, root)})
    request, cell, provenance = make_wrapper(root, 'core').authorized_build(isolated, run_id)
    def schema_request(case, alias, variant, credential):
        if variant not in schema.CONDITIONS:
            raise ValueError('Not a frozen schema condition')
        context.variant = variant
        try:
            return request(case, alias, variant, credential)
        finally:
            del context.variant
    cell.__globals__['request'] = schema_request
    return schema_request, cell, {**provenance, 'schema_sha256': SCHEMA_SHA,
        'wire_binding': 'Exact frozen insertion-order JSON bytes',
        'parser_binding': 'Private thread-local variant inside request worker'}


def selection_in_memory(target, run_id, schema, analysis):
    pilot, development = inventory(target, run_id, schema, True), inventory(target, run_id, schema)
    if not all(c['complete'] for c in pilot+development):
        raise ValueError('Incomplete pilot/development; no selection allowed')
    loader = types.FunctionType(schema.load.__code__, {**schema.load.__globals__, 'PROJECT': target},
                                argdefs=schema.load.__defaults__)
    dev = {(m, v, b): loader(m, v, b, 'prompt') for m, v, b in matrix(schema)}
    selected, cells = selected_matrix(schema, dev, analysis.metric)
    return dict(selected=selected, test_cells=cells, selection_written=False,
                test_used_for_selection=False)


def write_exclusive(path, value, canonical):
    with path.open('x') as out:
        out.write(canonical(value)+'\n'); out.flush(); os.fsync(out.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run_development(root, target, run_id, inference, schema, analysis, journal_dir):
    root, target, journal_dir = (Path(p).resolve() for p in (root, target, journal_dir))
    if type(run_id) is not int or run_id not in (2, 3) or target != replica_path(root, run_id):
        raise ValueError('Fresh replica identity required')
    if inference.PROJECT != target or schema.PROJECT != target:
        raise ValueError('Wrong schema workspace')
    source_check(schema)
    if (list(schema.MODELS.items()) != list(schema.protocol.MODELS.items())
            or schema.BENCHMARKS != schema.protocol.BENCHMARKS
            or schema.BINARY != BINARY or schema.CONDITIONS != BINARY+['C_review']):
        raise ValueError('Frozen schema matrix changed')
    if not journal_dir.is_relative_to(target/'artifacts') or journal_dir.exists():
        raise ValueError('Exclusive replica attempt required')
    progress = json.loads((target/'artifacts/progress.json').read_text())
    if progress.get('AP3_AP4_analysis', {}).get('status') != 'complete_automated':
        raise ValueError('Core prerequisite incomplete')
    protected = [target/'configs'/n for n in
                 ('expanded_schema_selection.json', 'expanded_schema_test_cells.json')]
    protected += [target/'artifacts'/n for n in ('expanded_schema_complete.json',
        'results/expanded_schema', 'schema_development_cost_forecast.json',
        'schema_confirmation_cost_forecast.json')]
    pilot_status = target/'artifacts/schema_pilot_status.json'
    if any(p.exists() for p in [*protected, pilot_status]):
        raise ValueError('Existing schema stage requires terminal review')
    manifest = schema.verify()
    if manifest['pilot']['families'] != PILOT_FAMILIES or manifest['pilot']['pairs_per_family'] != 2:
        raise ValueError('Frozen pilot changed')
    before = inventory(target, run_id, schema, True)+inventory(target, run_id, schema)
    # This entry admits only the first fresh attempt. Recovery is separate.
    if any((target/'runs'/ns).exists() and any((target/'runs'/ns).rglob('*'))
           for ns in ('schema_pilot', 'schema')):
        raise ValueError('Existing schema journals require terminal review')
    if any(c['expected'] != 2 for c in before[:len(matrix(schema, True))]):
        raise ValueError('Pilot must contain the frozen two pairs')
    _, cell, adapter = build_schema(root, schema, inference, run_id)
    journal_dir.mkdir(exist_ok=False)
    journal = journal_dir/'pilot.jsonl'
    write_exclusive(journal, dict(event='start', run_id=run_id,
        utc=datetime.now(timezone.utc).isoformat(), cell_order=matrix(schema, True),
        workers=2, invocations_per_cell=1), schema.wire)
    pilot_rows, pilot_complete = [], True
    for m, v, b in matrix(schema, True):
        try:
            cell(m, v, b, 'schema_pilot', workers=2, namespace='schema_pilot')
            checked = next(c for c in inventory(target, run_id, schema, True)
                           if (c['model'], c['variant'], c['benchmark']) == (m, v, b))
            if not checked['complete']:
                raise ValueError('Pilot cell incomplete')
            row = dict(model=m, condition=v, benchmark=b, status='passed',
                       known_cost_usd=checked['known_usd'], unknown_cost_attempts=checked['unknown_attempts'])
        except Exception as exc:
            pilot_complete = False
            row = dict(model=m, condition=v, benchmark=b, status='failed_requires_diagnosis',
                       error=str(exc), exception_type=type(exc).__name__)
        pilot_rows.append(row)
        durable_append(journal, dict(event='cell_end', **row), schema.wire)
        if not pilot_complete:
            break
    pilot_checked = inventory(target, run_id, schema, True)
    pilot_complete = pilot_complete and all(c['complete'] for c in pilot_checked)
    write_exclusive(pilot_status, dict(complete=pilot_complete, cells=pilot_rows), schema.protocol.canonical)
    durable_append(journal, dict(event='end', complete=pilot_complete), schema.wire)
    pilot_hash = hashlib.sha256(pilot_status.read_bytes()).hexdigest()
    outcomes = []
    if pilot_complete:
        def task(model):
            for m, v, b in matrix(schema):
                if m == model:
                    cell(m, v, b, 'prompt', workers=8, namespace='schema')
        outcomes = drain_model_tasks(task, list(schema.MODELS), journal_dir/'models.jsonl', schema.wire)
    checked = inventory(target, run_id, schema)
    complete = (pilot_complete and len(outcomes) == len(schema.MODELS)
                and all(m['status'] == 'complete' for m in outcomes)
                and all(c['complete'] for c in checked))
    selection = selection_in_memory(target, run_id, schema, analysis) if complete else None
    if any(p.exists() for p in protected) or hashlib.sha256(pilot_status.read_bytes()).hexdigest() != pilot_hash:
        raise ValueError('Unexpected selection/report/pilot writer')
    return dict(run_id=run_id, pilot_complete=pilot_complete, pilot_cells=pilot_checked,
        cells=checked, models=outcomes, development_complete=complete, selection_candidate=selection,
        adapter=adapter, selection_written=False, test_calls_started=False,
        admission_granted=False, report_written=False, completion_marker_written=False,
        historical_unjournaled_outcomes_resolved=False)
