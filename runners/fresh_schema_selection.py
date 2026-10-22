"""Durable own-run schema selection after the full pilot and development.

No CLI, admission, locks, test requests or final report. The future wrapper
must hold global shared and replica exclusive locks through all phases and
check real scheduler, prerequisite, scope, source, cost and budget gates.
Only a final fsynced receipt makes a freeze consumable; partial writes require
separate review. A consumer may see journals in exactly the selected test
scopes, but this component neither reads their outcomes nor certifies them.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import types

import fresh_schema_development_drained as development
from operational_journal import inspect_operations
from repeated_runs import replica_path, verify
from schema_drained import source_check
from schema_run_export import selected_matrix

DEVELOPMENT_SHA = 'fcb9457b889085de603bce23f41e603ecc1729a196454921f75e3578d7ea4293'
ATTEMPT = 'artifacts/fresh_schema_selection_freeze'
RECEIPT = ATTEMPT+'/committed.json'
OUTPUTS = ('configs/expanded_schema_selection.json', 'configs/expanded_schema_test_cells.json')
SUFFIXES = ('.jsonl', '.failures.jsonl', '.operations.jsonl', '.executions.jsonl')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def identity(root, target, run_id, schema):
    root, target = Path(root).resolve(), Path(target).resolve()
    if type(run_id) is not int or run_id not in (2, 3) or target != replica_path(root, run_id):
        raise ValueError('Fresh replica identity required')
    if schema.PROJECT != target:
        raise ValueError('Wrong schema workspace')
    verify(root, run_id)
    source_check(schema)
    if sha(Path(development.__file__)) != DEVELOPMENT_SHA:
        raise ValueError('Development component changed')
    if (list(schema.MODELS.items()) != list(schema.protocol.MODELS.items())
            or schema.BENCHMARKS != schema.protocol.BENCHMARKS
            or schema.BINARY != development.BINARY
            or schema.CONDITIONS != development.BINARY+['C_review']):
        raise ValueError('Frozen schema matrix changed')
    manifest = schema.verify()
    if (manifest['pilot']['families'] != development.PILOT_FAMILIES
            or manifest['pilot']['pairs_per_family'] != 2):
        raise ValueError('Frozen pilot changed')
    progress = json.loads((target/'artifacts/progress.json').read_text())
    if progress.get('AP3_AP4_analysis', {}).get('status') != 'complete_automated':
        raise ValueError('Core prerequisite incomplete')
    return root, target


def cell_paths(target, schema):
    result = []
    for pilot in (True, False):
        namespace, split = ('schema_pilot', 'schema_pilot') if pilot else ('schema', 'prompt')
        for m, v, b in development.matrix(schema, pilot):
            directory = target/'runs'/namespace/m/v/b
            result.append((m, v, b, pilot, target/'data'/b/(split+'.json'),
                           [directory/(split+s) for s in SUFFIXES]))
    return result


def own_path(target, path):
    path = Path(path)
    if not path.is_absolute():
        path = target/path
    if path.resolve() != path or not path.is_relative_to(target):
        raise ValueError('Foreign or symlinked schema provenance path')
    return path


def provenance(target, run_id, schema, model_journal):
    model_journal = own_path(target, model_journal)
    if not model_journal.is_relative_to(target/'artifacts') or model_journal.name != 'models.jsonl':
        raise ValueError('Own-run development model journal required')
    pilot_journal = model_journal.with_name('pilot.jsonl')
    paths = {model_journal, pilot_journal, target/'artifacts/schema_pilot_status.json'}
    events = [json.loads(s) for s in model_journal.read_text().splitlines()]
    ends = events[1:-1]
    if (not events or events[0].get('event') != 'start'
            or events[0].get('models') != list(schema.MODELS)
            or events[0].get('invocations_per_model') != 1
            or events[-1].get('event') != 'end' or events[-1].get('status') != 'complete'
            or events[-1].get('models_drained') != len(schema.MODELS)
            or len(ends) != len(schema.MODELS)
            or {e.get('model') for e in ends} != set(schema.MODELS)
            or any(e.get('event') != 'model_end' or e.get('status') != 'complete' for e in ends)):
        raise ValueError('All development model completions required')
    pilot = [json.loads(s) for s in pilot_journal.read_text().splitlines()]
    order = [list(c) for c in development.matrix(schema, True)]
    if (len(pilot) != len(order)+2 or pilot[0].get('event') != 'start'
            or type(pilot[0].get('run_id')) is not int or pilot[0]['run_id'] != run_id
            or pilot[0].get('workers') != 2 or pilot[0].get('invocations_per_cell') != 1
            or pilot[0].get('cell_order') != order
            or pilot[-1] != {'event': 'end', 'complete': True}
            or [[e.get('model'), e.get('condition'), e.get('benchmark')] for e in pilot[1:-1]] != order
            or any(e.get('event') != 'cell_end' or e.get('status') != 'passed' for e in pilot[1:-1])):
        raise ValueError('Complete ordered pilot journal required')
    status = json.loads((target/'artifacts/schema_pilot_status.json').read_text())
    if status != dict(complete=True, cells=[{k:v for k,v in e.items() if k != 'event'} for e in pilot[1:-1]]):
        raise ValueError('Pilot status differs from journal')
    for _, _, _, _, data, journals in cell_paths(target, schema):
        paths.update([data, *journals])
    return {str(p.relative_to(target)): sha(own_path(target, p)) for p in sorted(paths)}


def reconstruct(target, run_id, schema, analysis, allow_selected_tests=False):
    """Full raw pilot/Dev replay; test outcomes never influence selection."""
    expected, pilot_rows, groups = set(), [], {True: [], False: []}
    bound = types.SimpleNamespace(payload=schema.payload, digest=schema.wire_hash)
    for m, v, b, pilot, data, journals in cell_paths(target, schema):
        expected.update(journals)
        cases = json.loads(data.read_text())
        def rows(p):
            return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
        failures = rows(journals[1])
        checked = development.audit_cell(cases, rows(journals[0]), failures, schema, m, v, b, run_id)
        operations = inspect_operations(journals[2], cases, failures, bound, m, v, run_id)
        if operations['unknown_outcome_events']:
            raise ValueError('Unknown schema outcomes require terminal review')
        if not checked['complete'] or (pilot and len(cases) != 2):
            raise ValueError('Incomplete pilot/development; no selection allowed')
        groups[pilot].append(checked)
        if pilot:
            pilot_rows.append(dict(model=m, condition=v, benchmark=b, status='passed',
                known_cost_usd=checked['known_usd'], unknown_cost_attempts=checked['unknown_attempts']))
    if json.loads((target/'artifacts/schema_pilot_status.json').read_text()) != dict(complete=True, cells=pilot_rows):
        raise ValueError('Pilot status differs from raw replay')
    loader = types.FunctionType(schema.load.__code__, {**schema.load.__globals__, 'PROJECT': target},
                                argdefs=schema.load.__defaults__)
    dev = {(m,v,b):loader(m,v,b,'prompt') for m,v,b in development.matrix(schema)}
    selected, cells = selected_matrix(schema, dev, analysis.metric)
    if allow_selected_tests:
        # Permit only names implied by the rederived selection, not saved input.
        # Test contents/completeness must be audited by the test/report component.
        for m,v,b in cells:
            expected.update(target/'runs/schema'/m/v/b/('test'+s) for s in SUFFIXES)
    observed = set().union(*(set((target/'runs'/n).rglob('*.jsonl')) for n in ('schema','schema_pilot')))
    if observed-expected:
        raise ValueError('Unexpected schema journals require terminal review')
    for p in observed: own_path(target, p)
    return dict(selected=selected, test_cells=cells, pilot_cells=len(groups[True]),
        development_cells=len(groups[False]), pilot_responses=sum(c['responses'] for c in groups[True]),
        development_responses=sum(c['responses'] for c in groups[False]),
        selection_written=False, test_used_for_selection=False)


def implementation_sources(schema, analysis):
    files = {Path(__file__), Path(development.__file__), Path(schema.__file__),
             Path(analysis.metric.__code__.co_filename), Path(development.audit_rows.__code__.co_filename),
             Path(inspect_operations.__code__.co_filename), Path(selected_matrix.__code__.co_filename)}
    return {p.name:sha(p) for p in sorted(files)}


def output_bytes(candidate, schema):
    return {n:(schema.protocol.canonical(candidate[k])+'\n').encode()
            for n,k in zip(OUTPUTS, ('selected', 'test_cells'), strict=True)}


def durable_write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    sync_directory(path.parent)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def counts(candidate):
    return {k:candidate[k] for k in ('pilot_cells','development_cells','pilot_responses','development_responses')} | {
        'selected_models':len(candidate['selected']), 'unique_test_cells':len(candidate['test_cells'])}


def freeze_development(root, target, run_id, schema, analysis, model_journal):
    _, target = identity(root, target, run_id, schema)
    if any((target/p).exists() for p in [ATTEMPT, *OUTPUTS, 'artifacts/expanded_schema_complete.json',
                                       'artifacts/results/expanded_schema']):
        raise ValueError('Existing freeze or report requires separate review')
    before = provenance(target, run_id, schema, model_journal)
    sources = implementation_sources(schema, analysis)
    candidate = reconstruct(target, run_id, schema, analysis)
    if before != provenance(target, run_id, schema, model_journal) or sources != implementation_sources(schema, analysis):
        raise ValueError('Development changed during selection')
    rendered = output_bytes(candidate, schema)
    (target/ATTEMPT).mkdir()
    for name, raw in rendered.items(): durable_write(target/name, raw)
    for p in (target/ATTEMPT, target/'configs', target/'artifacts', target): sync_directory(p)
    if before != provenance(target, run_id, schema, model_journal) or sources != implementation_sources(schema, analysis):
        raise ValueError('Development changed during freeze; no commit')
    receipt = dict(run_id=run_id, committed_at=datetime.now(timezone.utc).isoformat(),
        status='schema_development_selection_frozen', model_journal=str(Path(model_journal).resolve().relative_to(target)),
        development_sha256=before, implementation_sha256=sources, component_sha256=sha(Path(__file__)), development_component_sha256=DEVELOPMENT_SHA,
        outputs_sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in rendered.items()},
        **counts(candidate), test_used_for_selection=False, admission_granted=False,
        test_calls_started=False, final_report_written=False)
    durable_write(target/RECEIPT, (schema.protocol.canonical(receipt)+'\n').encode())
    return receipt


def verify_frozen(root, target, run_id, schema, analysis):
    """Read-only replay usable before/after tests; does not validate test outcomes."""
    _, target = identity(root, target, run_id, schema)
    if not (target/RECEIPT).exists(): raise ValueError('No committed development freeze')
    receipt = json.loads(own_path(target,target/RECEIPT).read_text())
    if (type(receipt.get('run_id')) is not int or receipt['run_id'] != run_id
            or receipt.get('status') != 'schema_development_selection_frozen'
            or any(receipt.get(k) is not False for k in ('test_used_for_selection','admission_granted','test_calls_started','final_report_written'))
            or receipt.get('component_sha256') != sha(Path(__file__))
            or receipt.get('development_component_sha256') != DEVELOPMENT_SHA
            or receipt.get('implementation_sha256') != implementation_sources(schema, analysis)):
        raise ValueError('Invalid freeze identity or provenance')
    journal = own_path(target, receipt['model_journal'])
    before = provenance(target, run_id, schema, journal)
    if receipt['development_sha256'] != before: raise ValueError('Frozen development provenance changed')
    candidate = reconstruct(target, run_id, schema, analysis, allow_selected_tests=True)
    rendered = output_bytes(candidate, schema)
    if (receipt['outputs_sha256'] != {n:hashlib.sha256(raw).hexdigest() for n,raw in rendered.items()}
            or any(own_path(target,target/n).read_bytes() != raw for n,raw in rendered.items())
            or any(receipt.get(k) != v for k,v in counts(candidate).items())):
        raise ValueError('Frozen selection differs from full raw reconstruction')
    if (before != provenance(target, run_id, schema, journal)
            or receipt['implementation_sha256'] != implementation_sources(schema, analysis)):
        raise ValueError('Development changed during verification')
    return candidate
