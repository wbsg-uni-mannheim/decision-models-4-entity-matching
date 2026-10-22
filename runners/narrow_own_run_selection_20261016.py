"""Bounded raw-backed DEV selection/freeze; no inference or admission CLI.

Only original Runs 2/3. Reuses the frozen payloads, parsers and selector.
Does not claim scheduler/history/model-end/credit validation or permission to
start tests. The future runtime must establish those independently, and bind
this immutable receipt before test inference. No old selections are replaced.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import types

from diagnose_fresh_baseline import audit_rows
import narrow_core_scope_20261016 as scope
from operational_journal import inspect_operations
from repeated_runs import verify as verify_replica

HELPER_PINS = {
    'artifacts/operations/narrow_core_scope_20261016.py':
        'ad438185ade5387b29062e3dc975c7d716deb4e89838a47d06e04ecbfb99271f',
    'artifacts/operations/diagnose_fresh_baseline.py':
        '21fbb7417b987f3d12e775d5e0fa53b75f7e9ba8f1b02a1221c76d9b11ff7357',
    'artifacts/operations/operational_journal.py':
        '074439fc1a5f88a05cc075c60ac989ba403a86c8dad1a016d5f56e4789895a0f',
    'artifacts/operations/repeated_runs.py':
        'c4f0ba89bf9abf88f9675888654f6b594932d87bd9d728b69615a8d5d48818ab',
}
OUTPUT = 'artifacts/narrow_own_run_selection_20261016_v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def no_links(path, root):
    """No cache/output symlink, including ancestor links; root is resolved."""
    path.relative_to(root)
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError('Symlink in cache/output/source path')


class Snapshot:
    def __init__(self, root):
        self.root, self.states = root, {}

    def read(self, path, required=True, frozen_data=False):
        if frozen_data:
            relative = path.relative_to(self.root).parts
            if len(relative) != 5 or relative[:1] != ('replications',) or relative[2] != 'data':
                raise ValueError('Foreign frozen data path')
            expected = self.root / 'data' / relative[3] / relative[4]
            no_links(expected, self.root)
            if path.resolve() != expected:
                raise ValueError('Foreign frozen data path')
        else:
            no_links(path, self.root)
        raw = path.read_bytes() if path.exists() else None
        if raw is None and required:
            raise ValueError('Missing development source: ' + str(path))
        key = str(path.relative_to(self.root))
        value = sha(raw) if raw is not None else None
        if key in self.states and self.states[key] != value:
            raise ValueError('Source changed while selecting')
        self.states[key] = value
        return raw

    def recheck(self):
        for relative, expected in list(self.states.items()):
            path = self.root / relative
            self.read(path, required=expected is not None, frozen_data=relative.startswith(
                ('replications/run02/data/', 'replications/run03/data/')))


def identity(root, run_id, protocol, expanded, schema, confirmation, snapshot):
    scope.check_run(run_id)
    if run_id not in (2, 3):
        raise ValueError('Preserve original Run1 choices; fresh selection only Run2/3')
    target = root / 'replications' / f'run{run_id:02d}'
    no_links(target, root)
    scope.verify_sources(root)
    spec = verify_replica(root, run_id)
    for relative, expected in {**scope.PINS, **HELPER_PINS}.items():
        if sha(snapshot.read(root / relative)) != expected:
            raise ValueError('Changed selection source: ' + relative)
    snapshot.read(Path(__file__).resolve())
    snapshot.read(target / 'replica.json')
    for relative in spec['immutable_sha256']:
        snapshot.read(root / relative)
        snapshot.read(target / relative)
    for module, name in ((protocol, 'protocol'), (expanded, 'expanded_protocol'),
                         (schema, 'expanded_schema'), (confirmation, 'expanded_confirmation')):
        path = Path(module.__file__).resolve()
        if path != target / 'src' / (name + '.py') or sha(snapshot.read(path)) != scope.PINS['src/' + name + '.py']:
            raise ValueError('Wrong private frozen module workspace')
    if (schema.protocol is not protocol or confirmation.payload is not expanded.payload
            or tuple(protocol.MODELS) != scope.MODELS or tuple(protocol.BENCHMARKS) != scope.BENCHMARKS
            or tuple(confirmation.PROMPT_VARIANTS) != scope.PROMPTS
            or tuple(schema.CONDITIONS) != scope.SCHEMAS):
        raise ValueError('Wrong frozen payload/parser/matrix binding')
    # No screen/core/pilot completion is fabricated here. The new runtime's
    # separate history admission must address the actual original histories.
    return target


def read_cell(target, run_id, namespace, model, variant, benchmark, cases,
              protocol, expanded, schema, snapshot):
    directory = target / 'runs' / namespace / model / variant / benchmark
    records = {}
    for suffix in ('.jsonl', '.failures.jsonl', '.operations.jsonl', '.executions.jsonl'):
        raw = snapshot.read(directory / ('prompt' + suffix), required=False)
        records[suffix] = [json.loads(line) for line in raw.splitlines()] if raw else []
    successes, failures = records['.jsonl'], records['.failures.jsonl']
    if any(row['variant'] != variant for row in successes + failures):
        raise ValueError('Foreign development variant')
    if namespace == 'schema':
        adapter = types.SimpleNamespace(
            digest=lambda value: sha(value.encode()) if isinstance(value, str) else protocol.digest(value),
            payload=lambda case, alias, _: schema.wire(schema.payload(case, alias, variant)),
            parse_response=lambda alias, raw: schema.parse(alias, raw, variant))
        ops = types.SimpleNamespace(payload=schema.payload, digest=schema.wire_hash)
    else:
        adapter = types.SimpleNamespace(digest=protocol.digest,
            payload=lambda case, alias, _: expanded.payload(case, alias, variant),
            parse_response=protocol.parse_response)
        ops = types.SimpleNamespace(payload=expanded.payload, digest=protocol.digest)
    audit = audit_rows(cases, [{**r, 'variant': 'B0'} for r in successes],
                       [{**r, 'variant': 'B0'} for r in failures], adapter, model, benchmark, run_id)
    operations = inspect_operations(directory / 'prompt.operations.jsonl', cases, failures,
                                     ops, model, variant, run_id)
    if not audit['complete'] or operations['unknown_outcome_events']:
        raise ValueError('Incomplete or unrecorded development outcomes; no selection')
    labels = {c['pair_id']: c['label'] for c in cases}
    counts = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'review': 0}
    for row in successes:
        pred, gold = row['prediction'], labels[row['pair_id']]
        if pred == -1 and variant == 'C_review':
            counts['review'] += 1
        elif pred in (0, 1):
            counts['tp' if gold == pred == 1 else 'tn' if gold == pred == 0
                   else 'fp' if pred == 1 else 'fn'] += 1
        else:
            raise ValueError('Invalid development prediction')
    denominator = 2 * counts['tp'] + counts['fp'] + counts['fn']
    return dict(run_id=run_id, namespace=namespace, model=model, variant=variant,
                benchmark=benchmark, split='prompt', **audit, **counts,
                f1=2 * counts['tp'] / denominator if denominator else 0., operations=operations)


def choose(run_id, metrics, confirmation):
    """Uses the frozen selector; complete DEV scores, never a test metric."""
    scope.check_run(run_id)
    expected = set(scope.development_cells(run_id)) | {
        scope.cell(run_id, 'quality', m, 'B0', b, 'prompt')
        for m in scope.MODELS for b in scope.BENCHMARKS}
    indexed = {}
    for row in metrics:
        scope.check_run(row['run_id'])
        key = tuple(row[k] for k in ('run_id', 'namespace', 'model', 'variant', 'benchmark', 'split'))
        if key not in expected or key in indexed or not row['complete']:
            raise ValueError('Wrong, duplicate or incomplete development matrix')
        value = row['f1']
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid development F1')
        indexed[key] = value
    if set(indexed) != expected:
        raise ValueError('Full narrow development matrix required')
    groups = []
    for namespace, variants in (('quality', scope.PROMPTS), ('schema', scope.BINARY)):
        rows = []
        for model in scope.MODELS:
            scores = {(b, v): indexed[scope.cell(run_id, namespace, model, v, b, 'prompt')]
                      for b in scope.BENCHMARKS for v in variants}
            for benchmark in scope.BENCHMARKS:
                for regime in ('global', 'per_benchmark'):
                    sources = list(scope.BENCHMARKS) if regime == 'global' else [benchmark]
                    winner, means = confirmation.choose_variant(scores, variants, sources)
                    rows.append(dict(run_id=run_id, model=model, benchmark=benchmark,
                                     regime=regime, variant=winner, source_benchmarks=sources,
                                     development_macro_f1=means))
        groups.append(rows)
    cells = scope.selected_test_cells(run_id, *groups)
    return dict(run_id=run_id, prompt_choices=groups[0], binary_choices=groups[1],
                test_cells=[list(c) for c in sorted(cells)], selection_split='D_prompt',
                test_used_for_selection=False)


@contextmanager
def locks(root, run_id):
    paths = [(root / 'artifacts/operations/repetition_writer.lock', fcntl.LOCK_SH),
             (root / 'replications' / f'run{run_id:02d}' / 'artifacts/parallel_writer.lock', fcntl.LOCK_EX)]
    for path, _ in paths:
        no_links(path, root)
    with paths[0][0].open('a') as global_lock, paths[1][0].open('a') as replica_lock:
        for handle, (_, mode) in zip((global_lock, replica_lock), paths):
            no_links(Path(handle.name), root)
            fcntl.flock(handle, mode | fcntl.LOCK_NB)
        yield


def write_exclusive(path, value):
    raw = (canonical(value) + '\n').encode()
    with path.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    sync_dir(path.parent)
    if path.read_bytes() != raw:
        raise ValueError('Selection durable readback mismatch')
    return sha(raw)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def freeze(root, run_id, protocol, expanded, schema, confirmation):
    """Exclusive separate selection receipt; no admission, funding or test call.

    Intended for a separate terminal-development phase. Must not be called by
    a wrapper already holding the same replica lock. Partial outputs stay as
    diagnostic evidence and block another freeze; no rollback or overwrite.
    """
    scope.check_run(run_id)
    if run_id not in (2, 3):
        raise ValueError('Fresh selection only Run2/3')
    root = Path(root).resolve()
    with locks(root, run_id):
        snap = Snapshot(root)
        target = identity(root, run_id, protocol, expanded, schema, confirmation, snap)
        output = target / OUTPUT
        no_links(output, root)
        if output.exists():
            raise ValueError('Selection attempt already exists; no repeat')
        manifest = json.loads(snap.read(target / 'configs/manifest.json'))
        cases = {}
        for b in scope.BENCHMARKS:
            rows = json.loads(snap.read(target / 'data' / b / 'prompt.json', frozen_data=True))
            if (len(rows) != 200 or protocol.digest(rows) != manifest['benchmarks'][b]['splits']['prompt']['sha256']
                    or any(c['benchmark'] != b or type(c['label']) is not int
                           or c['label'] not in (0, 1) for c in rows)):
                raise ValueError('Frozen own-run D_prompt mismatch')
            cases[b] = rows
        matrix = set(scope.development_cells(run_id)) | {
            scope.cell(run_id, 'quality', m, 'B0', b, 'prompt')
            for m in scope.MODELS for b in scope.BENCHMARKS}
        metrics = [read_cell(target, run_id, ns, m, v, b, cases[b], protocol, expanded, schema, snap)
                   for _, ns, m, v, b, _ in sorted(matrix)]
        selection = choose(run_id, metrics, confirmation)
        snap.recheck()
        output.mkdir(exist_ok=False)
        sync_dir(output.parent)
        hashes = {name: write_exclusive(output / name, value) for name, value in
                  (('selection.json', selection), ('development_metrics.json', metrics),
                   ('sources.json', snap.states))}
        snap.recheck()
        receipt = dict(utc=datetime.now(timezone.utc).isoformat(), run_id=run_id,
                       status='own_run_dev_selection_frozen_not_admitted', outputs_sha256=hashes,
                       script_sha256=sha(Path(__file__).read_bytes()), development_cells=len(metrics),
                       test_used_for_selection=False, admission_granted=False,
                       history_worker_end_credit_validation='required_by_separate_runtime')
        write_exclusive(output / 'receipt.json', receipt)
        return receipt


def require_frozen(root, run_id):
    """Readback only; caller must hold runtime locks and separate admission.

    A receipt certifies the recorded selection, never a runnable job or a
    successful prior stage. Rehashes DEV sources including absent journals.
    """
    scope.check_run(run_id)
    if run_id not in (2, 3):
        raise ValueError('Fresh selection only Run2/3')
    root = Path(root).resolve()
    output = root / 'replications' / f'run{run_id:02d}' / OUTPUT
    snap = Snapshot(root)
    receipt = json.loads(snap.read(output / 'receipt.json'))
    if (type(receipt['run_id']) is not int or receipt['run_id'] != run_id
            or receipt['status'] != 'own_run_dev_selection_frozen_not_admitted'
            or receipt['admission_granted'] is not False or receipt['test_used_for_selection'] is not False
            or receipt['script_sha256'] != sha(Path(__file__).read_bytes())
            or set(receipt['outputs_sha256']) != {'selection.json', 'development_metrics.json', 'sources.json'}):
        raise ValueError('Wrong immutable selection receipt')
    outputs = {}
    for name, expected in receipt['outputs_sha256'].items():
        raw = snap.read(output / name)
        if sha(raw) != expected:
            raise ValueError('Frozen selection output changed')
        outputs[name] = json.loads(raw)
    selection = outputs['selection.json']
    if type(selection['run_id']) is not int or selection['run_id'] != run_id:
        raise ValueError('Foreign selection run')
    cells = scope.selected_test_cells(run_id, selection['prompt_choices'], selection['binary_choices'])
    if selection['test_cells'] != [list(c) for c in sorted(cells)]:
        raise ValueError('Selection test scope mismatch')
    sources = Snapshot(root)
    sources.states = outputs['sources.json']
    sources.recheck()
    snap.recheck()
    return selection
