"""Private fixed-condition repetition runtime, NOT a launcher or admission.

Only missing D_prompt answers of original Runs 2/3: unchanged 18 instructions,
S1 and five schemas. No selection, test, calibration, pilot or stage completion.
A future wrapper must verify actual histories/ALLsteps/ALLjoins, budget/scopes,
hold both locks, durably persist the prior BEFORE funding, and pass real gates.
This component accepts an already funded private credit, then reconciles only
its own durable delta. It never releases another credit or retries an attempt.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import types

from drained_inference import drain_model_tasks
from fresh_rules_grouped import prefix
from fresh_schema_selection import own_path, sha
from fresh_screen_drained import audit_cell as quality_audit
from fresh_schema_development_drained import audit_cell as schema_audit
from fresh_supplement_grouped import model_completions
from grouped_inference import build
import narrow_core_scope_20261016 as scope
from operational_journal import inspect_operations
from repeated_runs import verify

STAGE = 'same_condition_prompt_schema_repetition'
SUFFIXES = ('.jsonl', '.failures.jsonl', '.operations.jsonl', '.executions.jsonl')
PINS = {
    'fresh_rules_grouped': '150cb3f0d9f9c4767bc981484c1f47b76a3f29a36b93664bec16ca0978235297',
    'fresh_schema_selection': 'e301dc4299df965b29a1988c62890d40142e80505e38b2582ee0b6a9f8c38c6f',
    'drained_inference': '28e7ce2b2020a1d2b8a9def38c1b3e70e2b202938e5564a6a45cd981330a85b2',
    'narrow_core_scope_20261016': 'ad438185ade5387b29062e3dc975c7d716deb4e89838a47d06e04ecbfb99271f',
    'grouped_inference': '932c2e0bbd2342813b034887c3b38fd43735a32b830c1c983f4823438f5e7819',
    'leased_budget': 'f6ec051f1386db3696428e760d612e279ec64c95909fb3a3e649c8a54b44ad81',
    'fresh_screen_drained': 'b05dca3f0c4159eeb21f13cf2152bbef8d134e47ee3981355690c9ed3ce68956',
    'fresh_schema_development_drained': 'fcb9457b889085de603bce23f41e603ecc1729a196454921f75e3578d7ea4293',
    'fresh_supplement_grouped': '5db738b3f00c7bbf7085bc8d474f34f67be80d3b1d21ed6a4e4664c822bab33f',
    'diagnose_fresh_baseline': '21fbb7417b987f3d12e775d5e0fa53b75f7e9ba8f1b02a1221c76d9b11ff7357',
    'operational_journal': '074439fc1a5f88a05cc075c60ac989ba403a86c8dad1a016d5f56e4789895a0f',
    'repeated_runs': 'c4f0ba89bf9abf88f9675888654f6b594932d87bd9d728b69615a8d5d48818ab',
}


def matrix(rid):
    scope.check_run(rid)
    if rid not in (2, 3):
        raise ValueError('No repetition or recovery of original Run1')
    return [scope.cell(rid, ns, m, v, b, 'prompt') for m in scope.MODELS
            for ns, variants in (('quality', scope.PROMPTS + ('S1',)), ('schema', scope.SCHEMAS))
            for v in variants for b in scope.BENCHMARKS]


def sources(root, target, rid, inference, protocol, expanded, schema):
    root, target = Path(root).resolve(), Path(target)
    matrix(rid)
    if target != root / 'replications' / f'run{rid:02d}' or target.resolve() != target:
        raise ValueError('Exact own replica required')
    scope.verify_sources(root)
    spec = verify(root, rid)
    protected = {str(target / 'replica.json'): sha(target / 'replica.json')}
    for rel, expected in spec['immutable_sha256'].items():
        for base in (root, target):
            path = base / rel
            if sha(path) != expected:
                raise ValueError('Replica frozen source changed')
            protected[str(path)] = expected
    for name, expected in PINS.items():
        path = root / 'artifacts/operations' / (name + '.py')
        if sha(path) != expected:
            raise ValueError('Repetition helper changed: ' + name)
        protected[str(path)] = expected
    for rel, expected in scope.PINS.items():
        protected[str(root / rel)] = expected
    for module, name in ((inference, 'run'), (protocol, 'protocol'),
                         (expanded, 'expanded_protocol'), (schema, 'expanded_schema')):
        if Path(module.__file__).resolve() != target / 'src' / (name + '.py'):
            raise ValueError('Wrong private module workspace')
    if (inference.PROJECT != target or schema.PROJECT != target or schema.protocol is not protocol
            or tuple(protocol.MODELS) != scope.MODELS or tuple(protocol.BENCHMARKS) != scope.BENCHMARKS
            or tuple(schema.CONDITIONS) != scope.SCHEMAS):
        raise ValueError('Wrong private protocol bindings')
    manifest = json.loads((target / 'configs/manifest.json').read_text())
    cases = {}
    for b in scope.BENCHMARKS:
        p = target / 'data' / b / 'prompt.json'
        if p.resolve() != root / 'data' / b / 'prompt.json':
            raise ValueError('Foreign input symlink')
        rows = json.loads(p.read_text())
        if (len(rows) != 200 or protocol.digest(rows) != manifest['benchmarks'][b]['splits']['prompt']['sha256']
                or any(c['benchmark'] != b or type(c['label']) is not int or c['label'] not in (0, 1) for c in rows)):
            raise ValueError('Original fixed200 inputs changed')
        cases[b] = rows
        protected[str(p)] = sha(p)
    protected[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    return cases, protected


def paths(target, cells):
    result = set()
    for _, ns, m, v, b, split in cells:
        directory = target / 'runs' / ns / m / v / b
        allowed = {directory / (split + suffix) for suffix in SUFFIXES}
        if set(directory.glob(split + '*.jsonl')) - allowed:
            raise ValueError('Unreviewed repetition journals')
        for path in allowed:
            own_path(target, path)
        result.update(allowed)
    return sorted(result)


def rows(path):
    prefix(path)
    return [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []


def inventory(target, rid, cases, protocol, expanded, schema):
    result = []
    for cell in matrix(rid):
        _, ns, m, v, b, split = cell
        directory = target / 'runs' / ns / m / v / b
        paths(target, [cell])
        successes, failures = rows(directory / (split + '.jsonl')), rows(directory / (split + '.failures.jsonl'))
        if ns == 'quality':
            audit = quality_audit(cases[b], successes, failures, protocol, expanded.payload, m, v, b, rid)
            bound = types.SimpleNamespace(payload=expanded.payload, digest=protocol.digest)
        else:
            audit = schema_audit(cases[b], successes, failures, schema, m, v, b, rid)
            bound = types.SimpleNamespace(payload=schema.payload, digest=schema.wire_hash)
        operations = inspect_operations(directory / (split + '.operations.jsonl'), cases[b], failures,
                                         bound, m, v, rid)
        if operations['unknown_outcome_events']:
            raise ValueError('Unknown unrecorded outcome; keep credit')
        found = {r['pair_id'] for r in successes}
        missing = [c['pair_id'] for c in cases[b] if c['pair_id'] not in found]
        result.append(dict(cell=list(cell), **audit, operations=operations, missing_pair_ids=missing))
    return result


def unchanged(protected):
    if any(sha(Path(p)) != expected for p, expected in protected.items()):
        raise ValueError('Frozen sources changed')


def capture_prior(root, target, rid, inference, protocol, expanded, schema, journal_dir):
    """Read only. Caller must persist/fsync/readback before credit funding."""
    journal_dir = own_path(target, journal_dir)
    if not journal_dir.is_relative_to(target / 'artifacts') or journal_dir.exists():
        raise ValueError('Exclusive repetition journal required')
    cases, protected = sources(root, target, rid, inference, protocol, expanded, schema)
    journals = {str(p.relative_to(target)): prefix(p) for p in paths(target, matrix(rid))}
    checked = inventory(target, rid, cases, protocol, expanded, schema)
    if all(c['complete'] for c in checked):
        raise ValueError('All fixed conditions already complete; no repeat')
    for c in checked:
        if c['failure_rows'] or c['operations']['events']:
            if c['cell'][1] != 'quality' or c['cell'][3] != 'S1' or not c['complete']:
                raise ValueError('Historical failures require separate diagnosis, no blind retry')
    if journals != {p: prefix(target / p) for p in journals}:
        raise ValueError('Cache changed during prior audit')
    unchanged(protected)
    return dict(run_id=rid, stage=STAGE, replica=str(target), journal_dir=str(journal_dir.relative_to(target)),
                cells=checked, protected_sha256=protected, journals=journals,
                prior_responses=sum(c['responses'] for c in checked), admission_granted=False)


def read_prior(target, rid, prior_path, prior_sha, journal_dir):
    prior_path, journal_dir = own_path(target, prior_path), own_path(target, journal_dir)
    if not prior_path.is_relative_to(target / 'artifacts') or sha(prior_path) != prior_sha:
        raise ValueError('Durable prior changed')
    prior = json.loads(prior_path.read_text())
    if (type(prior['run_id']) is not int or prior['run_id'] != rid or prior['stage'] != STAGE
            or prior['replica'] != str(target) or prior['journal_dir'] != str(journal_dir.relative_to(target))
            or [c['cell'] for c in prior['cells']] != [list(c) for c in matrix(rid)]
            or set(prior['journals']) != {str(p.relative_to(target)) for p in paths(target, matrix(rid))}):
        raise ValueError('Wrong prior scope/run/attempt')
    unchanged(prior['protected_sha256'])
    return prior


def credit_identity(root, target, rid, prior_sha, journal_dir, credit):
    if credit.root.resolve() != root.resolve() or credit.chunk != 8 or credit.closed:
        raise ValueError('Wrong or closed credit')
    row = json.loads((root / 'artifacts/budget.json').read_text())['inference_credits'][credit.identity]
    p = row['provenance']
    if (row['status'] != 'active' or p.get('prior_sha256') != prior_sha or type(p.get('run_id')) is not int
            or p['run_id'] != rid or p.get('stage') != STAGE
            or p.get('journal_dir') != str(journal_dir.relative_to(target))):
        raise ValueError('Credit does not bind durable prior/run/attempt')


def execute(root, target, rid, inference, protocol, expanded, schema,
            prior_path, prior_sha, journal_dir, credit):
    """No funding or admission. Supplied credit must be fresh, scoped and real."""
    credit_identity(root, target, rid, prior_sha, journal_dir, credit)
    if credit.attempts or credit.settled:
        raise ValueError('Credit already used')
    prior = read_prior(target, rid, prior_path, prior_sha, journal_dir)
    if prior != capture_prior(root, target, rid, inference, protocol, expanded, schema, journal_dir):
        raise ValueError('Prior cache changed before execution')
    cases, _ = sources(root, target, rid, inference, protocol, expanded, schema)
    allowed = {tuple(c['cell']): c['missing_pair_ids'] for c in prior['cells']}
    case_index = {(b, c['pair_id']): c for b, block in cases.items() for c in block}
    allowed_cells = set(matrix(rid))
    seen, mutex, context = set(), threading.Lock(), threading.local()
    adapters = {}
    for namespace in ('quality', 'schema'):
        kwargs = dict(payload=expanded.payload, digest=protocol.digest,
                      canonical=protocol.canonical, parse_response=protocol.parse_response)
        if namespace == 'schema':
            kwargs.update(payload=schema.payload, digest=schema.wire_hash, canonical=schema.wire,
                          parse_response=lambda m, raw: schema.parse(m, raw, context.variant))
        isolated = types.SimpleNamespace(**{**vars(inference), **kwargs, 'account': credit.account,
            '_on_journal_failure': lambda: setattr(credit, 'journal_io_failed', True)})
        request, call, _ = build(isolated, rid)
        def guarded(case, model, variant, credential, request=request, ns=namespace):
            c = scope.cell(rid, ns, model, variant, case['benchmark'], 'prompt')
            key = (*c, case['pair_id'])
            if c not in allowed or case['pair_id'] not in allowed[c] or case != case_index.get((case['benchmark'], case['pair_id'])):
                raise ValueError('Request outside exact missing fixed-condition scope')
            with mutex:
                if key in seen:
                    raise ValueError('Duplicate request invocation')
                seen.add(key)
            if ns == 'schema':
                context.variant = variant
            try:
                return request(case, model, variant, credential)
            finally:
                if ns == 'schema':
                    del context.variant
        call.__globals__['request'] = guarded
        adapters[namespace] = call
    try:
        journal_dir.mkdir(exist_ok=False)
        sync_dir(journal_dir.parent)
        def task(model):
            for c in matrix(rid):
                _, ns, m, variant, b, split = c
                if m != model or not allowed[c]:
                    continue
                scope.validate_call(rid, c, allowed[c], allowed[c], allowed_cells)
                adapters[ns](m, variant, b, split, workers=8, namespace=ns)
        outcomes = drain_model_tasks(task, list(scope.MODELS), journal_dir / 'models.jsonl', protocol.canonical)
        sync_dir(journal_dir)
        checked = inventory(target, rid, cases, protocol, expanded, schema)
        unchanged(prior['protected_sha256'])
        return dict(models=outcomes, cells=checked,
                    inference_complete=all(c['complete'] for c in checked)
                    and all(c['status'] == 'complete' for c in outcomes),
                    selection_written=False, test_calls=0, admission_granted=False, report_written=False)
    except OSError:
        credit.journal_io_failed = True
        raise


def reconcile(root, target, rid, inference, protocol, expanded, schema,
              prior_path, prior_sha, journal_dir, credit):
    if getattr(credit, 'journal_io_failed', False):
        raise ValueError('Unconfirmed durability; keep credit')
    credit_identity(root, target, rid, prior_sha, journal_dir, credit)
    prior = read_prior(target, rid, prior_path, prior_sha, journal_dir)
    ends = model_completions(journal_dir, protocol)
    cases, protected = sources(root, target, rid, inference, protocol, expanded, schema)
    if protected != prior['protected_sha256']:
        raise ValueError('Sources changed before reconciliation')
    observed = {p: prefix(target / p) for p in prior['journals']}
    checked = inventory(target, rid, cases, protocol, expanded, schema)
    attempts = unknown = 0
    costs = []
    for rel, old in prior['journals'].items():
        path = own_path(target, rel)
        raw = path.read_bytes() if path.exists() else b''
        n = old['bytes']
        if (type(n) is not int or n < 0 or len(raw) < n or (old['exists'] and not path.exists())
                or hashlib.sha256(raw[:n]).hexdigest() != old['sha256']
                or (n and raw[n-1:n] != b'\n') or (raw and not raw.endswith(b'\n'))):
            raise ValueError('Prior durable prefix changed')
        if rel.endswith(('.operations.jsonl', '.executions.jsonl')):
            continue  # Operations fully joined and unknown outcomes rejected above.
        for line in raw[n:].splitlines():
            for a in json.loads(line)['attempts']:
                attempts += 1
                if a['cost'] is None:
                    unknown += 1
                else:
                    costs.append(a['cost'])
    charge = math.fsum(costs) + .05 * unknown
    if (credit.outstanding or credit.attempts != credit.settled or attempts != credit.settled
            or unknown != credit.unknown or not math.isclose(charge, credit.charged, rel_tol=0, abs_tol=1e-8)):
        raise ValueError('Unreconciled durable delta; keep credit')
    if (observed != {p: prefix(target / p) for p in prior['journals']}
            or ends != model_completions(journal_dir, protocol)):
        raise ValueError('Journals changed during reconciliation')
    unchanged(protected)
    credit_identity(root, target, rid, prior_sha, journal_dir, credit)
    credit.checkpoint(close=True)
    return dict(attempts=attempts, unknown=unknown, charged=charge, prior_responses=prior['prior_responses'],
                inference_complete=all(c['complete'] for c in checked) and ends['status'] == 'complete',
                admission_granted=False, report_written=False)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
