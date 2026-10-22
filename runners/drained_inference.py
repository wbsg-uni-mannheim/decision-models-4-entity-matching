"""Opt-in operational adapter; does not edit or install into the frozen runner.

Builds two functions from the hash-pinned source with explicit, counted edits.
The caller must separately obtain terminal diagnosis, writer lock and admission.
No CLI, job submission, retry of failed stages or budget override is provided.
Guarantee is ordinary exception draining, not crash/power-loss recovery. Historical
unjournaled outcomes cannot be reconstructed by this adapter.
"""
import ast
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path

FROZEN_RUN_SHA = 'e6467e3267a282dd3652ae04b1354c4f767628b06895c82dcb31ec78c5ad17ab'


def durable_append(path, value, canonical):
    with path.open('a') as out:
        out.write(canonical(value)+'\n')
        out.flush()
        os.fsync(out.fileno())


def substitute(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Frozen operational edit anchor mismatch')
    return source.replace(old, new, 1)


def drain_model_tasks(task, models, journal, canonical):
    """One invocation per model, collect every ordinary worker exception.

    Future admitted launchers can use this instead of list(pool.map(...)).
    No automatic retry. A new exclusive journal is mandatory. Its caller must
    retain the writer lock until this function returns, including failed tasks.
    """
    if not models or len(set(models)) != len(models):
        raise ValueError('Nonempty distinct models required')
    with journal.open('x') as handle:
        def log(event, **fields):
            handle.write(canonical({'utc': datetime.now(timezone.utc).isoformat(),
                                    'event': event, **fields})+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        log('start', models=list(models), invocations_per_model=1)
        outcomes = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as pool:
            futures = {pool.submit(task, model): model for model in models}
            for future in concurrent.futures.as_completed(futures):
                model = futures[future]
                try:
                    future.result()
                    outcome = {'model': model, 'status': 'complete'}
                except Exception as exc:
                    outcome = {'model': model, 'status': 'failed', 'exception_type': type(exc).__name__,
                               'budget_reservation_rejected': type(exc) is RuntimeError
                               and str(exc) == 'Project spending ceiling reached'}
                outcomes.append(outcome)
                log('model_end', **outcome)
        log('end', status='failed' if any(r['status'] == 'failed' for r in outcomes) else 'complete',
            models_drained=len(outcomes))
        return outcomes


def build(frozen, run_id):
    """Return isolated (request, run_cell, provenance), leaving module untouched.

    frozen.account may already be an admitted shared-root binding. This function
    never changes its cap or ledger. Scope/run/Slurm authorization is the future
    launcher's responsibility, not inferred from building these functions.
    """
    if type(run_id) is not int or run_id not in (1, 2, 3):
        raise ValueError('Invalid study run')
    raw = Path(frozen.__file__).read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_RUN_SHA:
        raise ValueError('Frozen inference source changed')
    source = raw.decode()
    functions = {n.name: ast.get_source_segment(source, n)
                 for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    request_source = substitute(functions['request'], "        account('reserve')",
        "        try:\n            account('reserve')\n        except Exception as exc:\n"
        "            return _stopped(case, alias, variant, attempts, exc, credential, 'reserve')")
    cell_source = substitute(functions['run_cell'],
        'futures = [executor.submit(request, c, alias, variant, credential) for c in batch]',
        'futures = {executor.submit(request, c, alias, variant, credential): c for c in batch}')
    cell_source = substitute(cell_source, 'result = future.result()',
        'result = _collect(future, futures[future], alias, variant, credential)')
    cell_source = substitute(cell_source,
        "                        with (directory/(split+'.failures.jsonl')).open('a') as out:\n"
        "                            out.write(canonical(result)+'\\n')",
        "                        _write_failure(directory, split, result)")
    cell_source = substitute(cell_source,
        "                        with path.open('a') as out:\n                            out.write(canonical(result)+'\\n')",
        "                        _append(path, result)")
    cell_source = substitute(cell_source,
        "            with (directory/(split+'.executions.jsonl')).open('a') as out:\n"
        "                out.write(canonical(execution)+'\\n')",
        "            _append(directory/(split+'.executions.jsonl'), execution)")
    namespace = dict(vars(frozen))

    def stopped(case, alias, variant, attempts, exc, credential, phase):
        # Reserve failures are not HTTP attempts. Preserve paid earlier retries
        # exactly once in failures.jsonl, and separate the operational event.
        ceiling = phase == 'reserve' and type(exc) is RuntimeError and str(exc) == 'Project spending ceiling reached'
        error = 'Project spending ceiling reached' if ceiling else 'Operational '+phase+' exception: '+type(exc).__name__
        return {'benchmark': case['benchmark'], 'pair_id': case['pair_id'],
                'input_hash': case['input_hash'], 'request_hash': frozen.digest(frozen.payload(case, alias, variant)),
                'model_alias': alias, 'variant': variant, 'study_run_id': run_id,
                'error': error, 'attempts': attempts,
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'operational_stop': {'phase': phase, 'exception_type': type(exc).__name__,
                    'budget_reservation_rejected': ceiling,
                    'outcome_unknown': not ceiling, 'previous_attempts_preserved': len(attempts)}}

    def collect(future, case, alias, variant, credential):
        try:
            return future.result()
        except Exception as exc:
            return stopped(case, alias, variant, [], exc, credential, 'future')

    def append(path, value):
        durable_append(path, value, frozen.canonical)

    def write_failure(directory, split, result):
        # Operational metadata excludes attempts, preventing double accounting.
        if result.get('operational_stop'):
            append(directory/(split+'.operations.jsonl'),
                   {k: v for k, v in result.items() if k != 'attempts'})
        if result['attempts']:
            append(directory/(split+'.failures.jsonl'), result)

    namespace.update(_stopped=stopped, _collect=collect, _append=append, _write_failure=write_failure)
    code = request_source+'\n\n'+cell_source+'\n'
    exec(compile(code, '<drained frozen inference>', 'exec'), namespace)
    generated_request = namespace['request']

    def request(*args, **kwargs):
        return {**generated_request(*args, **kwargs), 'study_run_id': run_id}

    namespace['request'] = request
    # Use the same executor and as_completed semantics as the frozen runner.
    assert namespace['concurrent'].futures is concurrent.futures
    return request, namespace['run_cell'], {
        'frozen_run_sha256': FROZEN_RUN_SHA,
        'generated_functions_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'study_run_id': run_id, 'admission_granted': False,
        'edits': ['reserve exception becomes operational stop with prior attempts',
                  'future-to-case mapping and drain of ordinary future exceptions',
                  'fsync response/failure/operational/execution records', 'tag returned rows with run ID']}
