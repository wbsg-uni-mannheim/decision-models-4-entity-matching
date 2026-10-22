"""Qwen3.8-27B (base model of PPLX v1.2), generative, default prompt B0, full test sets, run 1-3 (--run-id).

Operational copy of run_valsel_20261019.py. Differences: split 'test', model qwen38 only,
request = Luna generative B0 request with the Qwen model (src/qwen38_baseline.py), study run id 1-3
with its own namespace, approval, manifest and credit prefix. Workers, HTTP timeout and caps
come from Aaron's approval file.
"""
import argparse
import concurrent.futures
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
import signal
import socket
import threading
from pathlib import Path
import subprocess
import sys
import time
import types

NAMESPACE = 'qwen38_b0_run01_20261022'
ATTEMPT = 'artifacts/operations/qwen38_b0_run01_20261022'
APPROVAL = 'artifacts/operations/approval_qwen38_baseline_aaron_20261022.json'
MANIFEST = 'configs/qwen38_b0_run01_manifest_20261022.json'
# Aaron, 22 Oct: "Qwen jetzt parallel launchen". Only jobs of this runner block a start; the shared
# ledger lock and project cap keep the budget safe across runners.
JOB_NAME = 'em-qwen38-b0'
# The same Aaron message approved this run and the reference run; either approval may hold the ledger authority.
SHARED_AUTHORITY = 'artifacts/operations/approval_reference_argon_aaron_20261021.json'
CREDIT_PREFIX = 'qwen38-b0-run1-'
STOP = threading.Event()
LOCK = 'artifacts/operations/qwen38_baseline_writer.lock'
SCOPE = 'qwen38_b0_baseline_runs123'
SPLIT = 'test'
RUN_HOLD = .005
RUN_ID = 1
COMMIT_EVERY = 64
SOURCES = ['src/run.py', 'src/protocol.py', 'src/expanded_protocol.py', 'src/expanded_rules.py',
           'src/prompt12.py', 'src/inputs_study.py', 'src/qwen38_baseline.py', 'configs/learned_rules.json', 'configs/manifest.json',
           'artifacts/operations/drained_inference.py', 'artifacts/operations/leased_budget.py',
           'artifacts/operations/run_qwen38_baseline_20261022.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(p, value):
    p = Path(p)
    with p.open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    sync_dir(p.parent)
    require(read(p) == value, 'Durable readback mismatch')


def append_lines(path, rows, canonical):
    if not rows:
        return
    new = not path.exists()
    with path.open('a') as out:
        out.writelines(canonical(r)+'\n' for r in rows)
        out.flush(); os.fsync(out.fileno())
    if new:
        sync_dir(path.parent)


def modules(root):
    sys.path.insert(0, str(root/'src'))
    sys.path.insert(0, str(root/'artifacts/operations'))
    import run
    import protocol
    import qwen38_baseline as study
    study.register_models()
    import drained_inference
    import leased_budget
    require(Path(run.__file__).resolve() == (root/'src/run.py').resolve(), 'Foreign run module')
    return run, protocol, study, drained_inference, leased_budget


def matrix(protocol, study, models=None):
    return [(m, v, b) for m in (models or study.RUN_MODELS) for v in study.run_variants(m) for b in study.benchmarks(v)]


def templates(protocol, study, models, variants):
    empty = {'pair_id': 'template', 'records': {'left': {}, 'right': {}}}
    return protocol.digest({b: {m: {v: study.payload({**empty, 'benchmark': b}, m, v) for v in variants(m)}
                                for m in models} for b in protocol.BENCHMARKS})


def install_dns_cache(ttl=300):
    """Process-wide getaddrinfo cache; request bytes and endpoints stay unchanged."""
    original = socket.getaddrinfo
    cache, lock, stats = {}, threading.Lock(), {'hits': 0, 'misses': 0, 'slow_lookups': 0}

    def cached(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        with lock:
            hit = cache.get(key)
            if hit and time.monotonic()-hit[0] < ttl:
                stats['hits'] += 1
                return hit[1]
        start = time.monotonic()
        value = original(*args, **kwargs)
        with lock:
            stats['misses'] += 1; stats['slow_lookups'] += time.monotonic()-start > 1
            cache[key] = (time.monotonic(), value)
        return value
    socket.getaddrinfo = cached
    return stats


# ---------------------------------------------------------------- authority

def approval(root):
    a = read(root/APPROVAL)
    require(a.get('approved_by') == 'Aaron' and a.get('scope') == SCOPE, 'Explicit Qwen3.8 baseline approval missing')
    require(type(a.get('project_cap_usd')) in (int, float) and type(a.get('run_cap_usd')) in (int, float)
            and 0 < a['run_cap_usd'] <= a['project_cap_usd'], 'Invalid approved caps')
    require(type(a.get('workers')) is int and 1 <= a['workers'] <= 32, 'Invalid approved worker count')
    require(type(a.get('timeout_s')) is int and 10 <= a['timeout_s'] <= 120, 'Invalid approved HTTP timeout')
    require(isinstance(a.get('quote'), str) and a['quote'].strip(), 'Approval quote required')
    return a


def run_remaining(root, a):
    """run_cap_usd covers the whole run: earlier jobs' credits of this run count against it."""
    credits = read(root/'artifacts/budget.json').get('inference_credits', {})
    used = sum(c['charged'] for k, c in credits.items() if k.startswith(CREDIT_PREFIX))
    return a['run_cap_usd']-used


def timeout_shim(seconds):
    """Stands in for the urllib module inside the isolated runner namespace."""
    import urllib.error
    import urllib.request

    def urlopen(req, timeout=None):
        return urllib.request.urlopen(req, timeout=seconds)
    return types.SimpleNamespace(request=types.SimpleNamespace(Request=urllib.request.Request, urlopen=urlopen),
                                 error=urllib.error)


def authorities(root, a):
    shared = read(root/SHARED_AUTHORITY)
    require(shared['quote'] == a['quote'] and shared['project_cap_usd'] == a['project_cap_usd'], 'Shared authority differs')
    return {sha(root/APPROVAL), sha(root/SHARED_AUTHORITY)}


def budget_check(root, a, headroom=0.):
    b = read(root/'artifacts/budget.json')
    require(b.get('project_cap_usd') == a['project_cap_usd'] and b.get('budget_authority_sha256') in authorities(root, a),
            'Ledger cap is not the approved baseline-run authority; run authorize-budget')
    require(all(type(b[k]) in (int, float) and math.isfinite(b[k]) and b[k] >= 0 for k in ('spent', 'reserved')), 'Invalid ledger')
    require(not any(x['active'] for x in b.get('maintenance_barriers', {}).values()), 'Maintenance barrier')
    require(b['spent']+b['reserved']+headroom <= a['project_cap_usd']+1e-8, 'Budget exhausted')
    return b


def authorize_budget(root):
    """Only raises the ledger ceiling to the approved value. Spent/reserved untouched."""
    _, _, _, _, leased_budget = modules(root)
    a = approval(root)
    require(not other_project_jobs(root), 'A project job is running; its credit checks the current authority')
    with leased_budget.ledger(root) as obj:
        old = obj.get('project_cap_usd')
        require(type(old) in (int, float) and a['project_cap_usd'] >= old, 'Cap may only increase')
        obj.setdefault('budget_authority_history', []).append(
            {'utc': now(), 'previous_cap_usd': old, 'previous_authority_sha256': obj.get('budget_authority_sha256'),
             'new_cap_usd': a['project_cap_usd'], 'approval': APPROVAL, 'approval_sha256': sha(root/APPROVAL)})
        obj['project_cap_usd'] = a['project_cap_usd']
        obj['budget_authority_sha256'] = sha(root/APPROVAL)
    return read(root/'artifacts/budget.json')['budget_authority_history'][-1]


def credit_class(root, leased_budget, a):
    """Credit with the approved cap; the original 200-dollar module stays untouched."""
    cap = a['project_cap_usd']
    allowed = authorities(root, a)

    @contextmanager
    def checked_ledger(directory):
        require(Path(directory).resolve() == root.resolve(), 'Foreign ledger')
        with leased_budget.ledger(directory) as obj:
            require(obj.get('project_cap_usd') == cap and obj.get('budget_authority_sha256') in allowed, 'Ledger policy drift')
            yield obj
    namespace = {**vars(leased_budget), 'CAP': cap, 'ledger': checked_ledger}
    methods = {}
    for name, function in vars(leased_budget.Credit).items():
        if isinstance(function, types.FunctionType):
            defaults = (0, cap) if name == 'account' else function.__defaults__
            methods[name] = types.FunctionType(function.__code__, namespace, function.__name__, defaults, function.__closure__)
    return type('Qwen38BaselineCredit', (), methods)


# ------------------------------------------------------------------ freeze

def freeze(root):
    """Freeze the run manifest; the test splits come from configs/manifest.json."""
    run, protocol, study, _, _ = modules(root)
    manifest = read(root/'configs/manifest.json')
    cells = matrix(protocol, study)
    splits = {b: manifest['benchmarks'][b]['splits']['test'] for b in protocol.BENCHMARKS}
    spec = {'scope': SCOPE, 'namespace': NAMESPACE, 'run_id': RUN_ID, 'split': SPLIT,
            'models': {m: protocol.MODELS[m] for m in study.RUN_MODELS},
            'variants': {'B0': {'name': 'Default prompt B0', 'chapter': 'baseline'}},
            'reference': 'Luna generative B0 request with qwen/qwen3.8-27b on deepinfra/bf16',
            'benchmarks': splits, 'cells': len(cells), 'decisions': sum(splits[b]['n'] for _, _, b in cells),
            'source_sha256': {rel: sha(root/rel) for rel in SOURCES},
            'request_templates_sha256': templates(protocol, study, study.RUN_MODELS, study.run_variants),
            'created_at': now()}
    save(root/MANIFEST, spec)
    return spec


def source_check(root, spec):
    for rel, digest in spec['source_sha256'].items():
        p = root/rel
        require(p.exists() and not p.is_symlink() and sha(p) == digest, 'Source drift: '+rel)


def inputs_check(root, protocol, spec):
    for b, info in spec['benchmarks'].items():
        cases = read(root/'data'/b/(SPLIT+'.json'))
        require(protocol.digest(cases) == info['sha256'] and len(cases) == info['n'], 'Frozen input mismatch: '+b)


def other_project_jobs(root, own=None):
    out = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i|%T|%j'], text=True)
    running = []
    for line in out.splitlines():
        job, state, name = line.split('|', 2)
        if job == own or name != JOB_NAME:
            continue
        meta = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
        if str(root) in meta and state != 'PENDING':
            running.append({'job': job, 'state': state, 'name': name})
    return running


def preflight(root, require_approval=True):
    run, protocol, study, _, _ = modules(root)
    spec = read(root/MANIFEST)
    source_check(root, spec)
    inputs_check(root, protocol, spec)
    checks = {'utc': now(), 'manifest_sha256': sha(root/MANIFEST),
              'namespace_exists': (root/'runs'/NAMESPACE).exists(), 'attempt_exists': (root/ATTEMPT).exists(),
              'other_running_project_jobs': other_project_jobs(root, os.environ.get('SLURM_JOB_ID'))}
    if require_approval:
        a = approval(root)
        checks['run_remaining_usd'] = run_remaining(root, a)
        checks['budget'] = budget_check(root, a, checks['run_remaining_usd'])
    return checks


# ----------------------------------------------------------------- execute

class CellFailed(RuntimeError):
    pass


def make_cell(root, protocol, study, request, workers):
    canonical = protocol.canonical

    def cell(alias, variant, benchmark, credential):
        cases = read(root/'data'/benchmark/(SPLIT+'.json'))
        directory = root/'runs'/NAMESPACE/alias/variant/benchmark
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/(SPLIT+'.jsonl')
        with (directory/(SPLIT+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(not (directory/(SPLIT+'.failures.jsonl')).exists() and not (directory/(SPLIT+'.operations.jsonl')).exists(),
                    'Earlier failure in cell; diagnose before resume')
            wanted = {c['pair_id']: c for c in cases}
            existing = {}
            if path.exists():
                for line in path.read_text().splitlines():
                    r = json.loads(line)
                    require(r['pair_id'] in wanted and r['pair_id'] not in existing, 'Foreign or duplicate cached row')
                    require(r['study_run_id'] == RUN_ID and r['request_hash'] == protocol.digest(study.payload(wanted[r['pair_id']], alias, variant)),
                            'Cache configuration mismatch')
                    existing[r['pair_id']] = r
            pending = iter([c for c in cases if c['pair_id'] not in existing])
            n_pending = len(cases)-len(existing)
            max_failures = max(20, len(cases)//100)  # sporadic HTTP 520 or invalid JSON must not stop a whole cell
            buffer, failures, inflight = [], [], {}
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                def submit():
                    c = next(pending, None)
                    if c is not None:
                        inflight[pool.submit(request, c, alias, variant, credential)] = c
                for _ in range(workers):
                    submit()
                while inflight:
                    done, _ = concurrent.futures.wait(inflight, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        c = inflight.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = {'benchmark': benchmark, 'pair_id': c['pair_id'], 'input_hash': c['input_hash'],
                                      'model_alias': alias, 'variant': variant, 'study_run_id': RUN_ID, 'attempts': [],
                                      'error': 'Operational future exception: '+type(exc).__name__, 'completed_at': now(),
                                      'operational_stop': {'phase': 'future', 'exception_type': type(exc).__name__, 'outcome_unknown': True}}
                        if result.get('error'):
                            failures.append(result)
                        else:
                            buffer.append(result)
                            existing[result['pair_id']] = result
                        if len(failures) <= max_failures and not STOP.is_set():
                            submit()  # isolated failures: keep going; outage-like bursts or SIGTERM: drain only
                    if len(buffer) >= COMMIT_EVERY:
                        append_lines(path, buffer, canonical); buffer = []
            append_lines(path, buffer, canonical)
            append_lines(directory/(SPLIT+'.operations.jsonl'), [{k: v for k, v in f.items() if k != 'attempts'}
                                                              for f in failures if f.get('operational_stop')], canonical)
            append_lines(directory/(SPLIT+'.failures.jsonl'), [f for f in failures if f['attempts']], canonical)
            elapsed = time.perf_counter()-start
            if n_pending:
                append_lines(directory/(SPLIT+'.executions.jsonl'), [{
                    'new_pairs': n_pending, 'completed_pairs': n_pending-len(failures) if not failures else None,
                    'wall_seconds': elapsed, 'workers': workers, 'window': 'sliding',
                    'throughput_pairs_per_second': n_pending/elapsed, 'completed_at': now()}], canonical)
            print('CELL', alias, variant, benchmark, len(existing), '/', len(cases), round(elapsed, 1), flush=True)
            if failures:
                raise CellFailed(f'{alias}/{variant}/{benchmark}: {len(failures)} failed request(s)')
            if STOP.is_set():
                raise RuntimeError('Stopped by SIGTERM; journals committed, resume with --resume')
            require(wanted.keys() == existing.keys(), 'Incomplete cell')
    return cell


def journal_totals(root, protocol, study, models=None):
    """Attempts already in the raw journals (non-zero only when resuming)."""
    totals = {'attempts': 0, 'unknown': 0, 'charged': 0.}
    for m, v, b in matrix(protocol, study, models):
        path = root/'runs'/NAMESPACE/m/v/b/(SPLIT+'.jsonl')
        for line in (path.read_text().splitlines() if path.exists() else []):
            for a in json.loads(line)['attempts']:
                totals['attempts'] += 1; totals['unknown'] += a['cost'] is None
                totals['charged'] += a['cost'] or 0.
    return totals


def validate(root, protocol, study, credit, unknown, base, models=None):
    totals = {'attempts': 0, 'unknown': 0, 'charged': 0.}
    for m, v, b in matrix(protocol, study, models):
        cases = {c['pair_id']: c for c in read(root/'data'/b/(SPLIT+'.json'))}
        folder = root/'runs'/NAMESPACE/m/v/b
        rows = [json.loads(x) for x in (folder/(SPLIT+'.jsonl')).read_text().splitlines()]
        require(len(rows) == len(cases) and {r['pair_id'] for r in rows} == cases.keys(), 'Incomplete/duplicate cell')
        for name in (SPLIT+'.failures.jsonl', SPLIT+'.operations.jsonl'):
            require(not (folder/name).exists(), 'Unresolved failure journal')
        for r in rows:
            c = cases[r['pair_id']]
            require(r['study_run_id'] == RUN_ID and r['model_alias'] == m and r['variant'] == v
                    and r['input_hash'] == c['input_hash']
                    and r['request_hash'] == protocol.digest(study.payload(c, m, v)), 'Raw identity mismatch')
            require(protocol.parse_response(m, r['attempts'][-1]['response']) == (r['prediction'], r['probability_match']),
                    'Raw parser mismatch')
            for a in r['attempts']:
                totals['attempts'] += 1; totals['unknown'] += a['cost'] is None
                totals['charged'] += a['cost'] or 0.
    require(credit.attempts == credit.settled == totals['attempts']-base['attempts']
            and unknown['count'] == totals['unknown']-base['unknown'] and credit.outstanding == 0
            and math.isclose(credit.charged, totals['charged']-base['charged'], abs_tol=1e-6),
            'Durable accounting mismatch')
    return totals


def key_usage(credential):
    """OpenRouter usage of the key in USD; best effort, never blocks the run."""
    import urllib.request
    try:
        req = urllib.request.Request('https://openrouter.ai/api/v1/key', headers={'Authorization': 'Bearer '+credential})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.load(response)['data']
        return {'utc': now(), 'usage_usd': data.get('usage'), 'limit_usd': data.get('limit')}
    except Exception as exc:
        return {'utc': now(), 'error': type(exc).__name__}


@contextmanager
def locks(root):
    with ExitStack() as stack:
        shared = stack.enter_context((root/'artifacts/operations/repetition_writer.lock').open('a'))
        fcntl.flock(shared, fcntl.LOCK_SH | fcntl.LOCK_NB)
        own = stack.enter_context((root/LOCK).open('a'))
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def execute(root, resume=False, models=None):
    job = os.environ.get('SLURM_JOB_ID')
    require(job, 'Execution only inside a Slurm job')
    run, protocol, study, drained_inference, leased_budget = modules(root)
    study.load_selection(root)
    a = approval(root)
    with locks(root):
        checks = preflight(root)
        require(not checks['other_running_project_jobs'], 'Other project jobs are running')
        require(resume or not checks['namespace_exists'], 'Namespace exists; pass --resume after diagnosis')
        attempt = root/ATTEMPT/('job-'+job)
        attempt.mkdir(parents=True); sync_dir(attempt.parent)
        models = models or list(study.RUN_MODELS)
        require(set(models) <= set(study.RUN_MODELS), 'Unknown or retired model')
        base = journal_totals(root, protocol, study, models)
        save(attempt/'prior.json', {**checks, 'journal_totals_before': base, 'models': models})
        credit = credit_class(root, leased_budget, a)(root, f'{CREDIT_PREFIX}{job}',
            {'run_id': RUN_ID, 'namespace': NAMESPACE, 'approval_sha256': sha(root/APPROVAL),
             'manifest_sha256': sha(root/MANIFEST), 'prior_sha256': sha(attempt/'prior.json')}, chunk=8)
        unknown = {'count': 0}
        remaining = checks['run_remaining_usd']
        try:
            def account(action, amount=0):
                with credit.lock:
                    if action == 'reserve' and credit.charged+RUN_HOLD*(credit.outstanding+1) > remaining:
                        raise RuntimeError('Project spending ceiling reached')
                    if action == 'settle' and amount is None:
                        unknown['count'] += 1  # no flat charge; counted, reconciled via key usage
                        amount = 0.
                    return credit.account(action, amount)
            pace = {'workers': a['workers'], 'timeout_s': a['timeout_s'], 'source': APPROVAL}
            isolated = types.SimpleNamespace(**{**vars(run), 'account': account, 'payload': study.payload,
                                                'urllib': timeout_shim(pace['timeout_s'])})
            request, _, binding = drained_inference.build(isolated, RUN_ID)
            cell = make_cell(root, protocol, study, request, pace['workers'])
            credential = run.key()
            dns = install_dns_cache()
            signal.signal(signal.SIGTERM, lambda *_: STOP.set())
            save(attempt/'started.json', {'utc': now(), 'binding': binding, 'workers': pace['workers'],
                                          'http_timeout_s': pace['timeout_s'], 'speed_source': pace['source'],
                                          'commit_every': COMMIT_EVERY, 'credit': credit.summary(),
                                          'unknown_cost_policy': 'counted, booked at 0 USD', 'models': models,
                                          'openrouter_key_usage_before': key_usage(credential)})

            def task(model):
                failed = []
                for v in study.run_variants(model):
                    for b in study.benchmarks(v):
                        if STOP.is_set():
                            raise RuntimeError('Stopped by SIGTERM before cell')
                        try:
                            cell(model, v, b, credential)
                        except CellFailed as exc:
                            failed.append(str(exc))
                            print('CELL_FAILED', exc, flush=True)
                if failed:
                    raise RuntimeError(f'{model}: {len(failed)} cell(s) with failed pairs; archive-failures, then resume')
            outcomes = drained_inference.drain_model_tasks(task, models, attempt/'models.jsonl', protocol.canonical)
            require(all(o['status'] == 'complete' for o in outcomes), 'At least one model stopped; credit retained')
            totals = validate(root, protocol, study, credit, unknown, base, models)
            save(attempt/'validation.json', {**totals, 'unknown_cost_attempts_this_job': unknown['count'],
                                             'models': models, 'dns_cache': dict(dns)})
            save(attempt/'credit_closed.json', credit.checkpoint(close=True))
            save(attempt/'complete.json', {'utc': now(), 'validation_sha256': sha(attempt/'validation.json'),
                                           'openrouter_key_usage_after': key_usage(credential)})
        except Exception as exc:
            if not credit.closed and not credit.outstanding:
                save(attempt/'credit_closed.json', credit.checkpoint(close=True))  # every attempt settled: exact
            save(attempt/'failed.json', {'utc': now(), 'exception_type': type(exc).__name__, 'message': str(exc)[:500],
                                         'credit': credit.summary(), 'unknown_cost_attempts': unknown['count'],
                                         'stopped_by_sigterm': STOP.is_set(),
                                         'hold_retained': not credit.closed})
            raise


def close_stale_credit(root, job):
    """Close the credit of a cancelled job against its durable journals.

    Responses buffered but not yet committed when the job was killed are lost;
    their cost is not in the journals and stays unknown (at most COMMIT_EVERY-1
    per active cell). The journal-based charge is therefore a lower bound.
    """
    _, protocol, study, _, leased_budget = modules(root)
    a = approval(root)
    raw = subprocess.check_output(['sacct', '-n', '-P', '-X', '-j', job, '--format=State'], text=True).split()
    require(raw and raw[0].split()[0] in ('CANCELLED', 'FAILED', 'TIMEOUT', 'COMPLETED', 'NODE_FAIL', 'OUT_OF_MEMORY'), 'Job not terminal')
    identity = f'{CREDIT_PREFIX}{job}'
    row = read(root/'artifacts/budget.json')['inference_credits'][identity]
    start = datetime.fromisoformat(row['created_at'])
    totals = {'attempts': 0, 'unknown': 0, 'charged': 0., 'records': 0}
    for path in (root/'runs'/NAMESPACE).rglob(SPLIT+'*.jsonl'):
        if path.name != SPLIT+'.jsonl' and not path.name.startswith(SPLIT+'.failures'):
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if datetime.fromisoformat(r['completed_at']) < start:
                continue
            totals['records'] += 1
            for att in r['attempts']:
                totals['attempts'] += 1; totals['unknown'] += att['cost'] is None
                totals['charged'] += att['cost'] or 0.
    with leased_budget.ledger(root) as obj:
        require(obj.get('project_cap_usd') == a['project_cap_usd'], 'Ledger cap drift')
        row = obj['inference_credits'][identity]
        require(row['status'] == 'active', 'Credit not active')
        committed = row['charged']
        obj['spent'] += totals['charged']-committed
        obj['reserved'] -= row['funded']-committed
        obj['attempts'] += totals['attempts']-row['attempts']
        row.update(status='closed_stale', charged=totals['charged'], attempts=totals['attempts'],
                   settled=totals['attempts'], unknown=totals['unknown'], closed_at=now(),
                   reconciliation={'basis': 'durable journals since credit creation', 'job': job,
                                   'committed_before': committed, 'records': totals['records'],
                                   'lost_buffered_responses': 'unknown, lower-bound charge'})
    return {'identity': identity, **totals, 'committed_before': committed}


def archive_failures(root):
    """Rename failure/operations journals so a resume re-requests the missing pairs.

    The archived files stay next to the cell for provenance and cost accounting.
    """
    require(not other_project_jobs(root), 'A project job is running')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    moved = []
    for name in (SPLIT+'.failures.jsonl', SPLIT+'.operations.jsonl'):
        for path in sorted((root/'runs'/NAMESPACE).rglob(name)):
            target = path.with_name(name.replace('.jsonl', f'.archived-{stamp}.jsonl'))
            require(not target.exists(), 'Archive target exists')
            rows = len(path.read_text().splitlines())
            os.rename(path, target); sync_dir(path.parent)
            moved.append({'from': str(path.relative_to(root)), 'to': target.name, 'rows': rows})
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--action', choices=['freeze', 'preflight', 'authorize-budget', 'execute',
                                         'close-stale-credit', 'archive-failures'], required=True)
    ap.add_argument('--job', help='Slurm job id for close-stale-credit')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--models', nargs='+', help='restrict execute to these model aliases')
    ap.add_argument('--run-id', type=int, choices=[1, 2, 3], required=True)
    args = ap.parse_args(); root = args.root.resolve()
    n = f'{args.run_id:02d}'
    globals().update(RUN_ID=args.run_id, NAMESPACE=f'qwen38_b0_run{n}_20261022',
                     ATTEMPT=f'artifacts/operations/qwen38_b0_run{n}_20261022',
                     MANIFEST=f'configs/qwen38_b0_run{n}_manifest_20261022.json', CREDIT_PREFIX=f'qwen38-b0-run{args.run_id}-')
    if args.action == 'freeze':
        print(json.dumps({k: v for k, v in freeze(root).items() if k != 'models'}, indent=1))
    elif args.action == 'preflight':
        print(json.dumps(preflight(root, require_approval=(root/APPROVAL).exists()), indent=1, default=str))
    elif args.action == 'authorize-budget':
        print(json.dumps(authorize_budget(root), indent=1))
    elif args.action == 'archive-failures':
        print(json.dumps(archive_failures(root), indent=1))
    elif args.action == 'close-stale-credit':
        print(json.dumps(close_stale_credit(root, args.job), indent=1))
    else:
        execute(root, args.resume, args.models)


if __name__ == '__main__':
    main()
