"""Admitted Peeters stage for each study run; frozen inference sources stay intact.

Preparation verifies every prospective payload without network or response writes.
Execution needs an actual Slurm admission and uses the shared atomic USD100 ledger.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import peeters_control as control
from parallel_admission import review
from repeated_runs import replica_path, sha, shared_account, verify as verify_replica

MANIFEST = 'configs/peeters_payload_manifest_20261014.json'
EXECUTION = 'configs/peeters_execution_manifest_20261014.json'
SOURCES = ['artifacts/operations/peeters_run.py', 'artifacts/operations/peeters_control.py',
           'artifacts/operations/parallel_admission.py', 'artifacts/operations/repeated_runs.py',
           'src/run.py', 'src/protocol.py', 'src/orchestrate.py', 'src/billiger50.py',
           'src/prepare.py', 'src/analyze.py']
NAMESPACE = 'peeters'
WORKERS = 8


def target_path(root, run_id):
    if run_id not in (1, 2, 3):
        raise ValueError('Unknown study run')
    return root if run_id == 1 else replica_path(root, run_id)


def load_modules(target):
    if any(k in sys.modules for k in ['run', 'protocol', 'orchestrate', 'billiger50']):
        raise RuntimeError('Fresh Python process required')
    sys.path.insert(0, str(target/'src'))
    import run
    import protocol
    import orchestrate
    import billiger50
    if any(m.PROJECT != target for m in [run, orchestrate, billiger50]):
        raise ValueError('Wrong source workspace')
    return run, protocol, orchestrate, billiger50


def verify_matrix(root, target, manifest, protocol):
    """Full matrix identity, inputs and ordered bodies, before any inference."""
    if manifest['question'] != control.QUESTION or manifest['arms'] != list(control.ARMS):
        raise ValueError('Question or arm drift')
    if sha(root/'artifacts/operations/peeters_control.py') != manifest['implementation_sha256']:
        raise ValueError('Payload builder drift')
    if sha(target/'src/protocol.py') != manifest['protocol_sha256']:
        raise ValueError('Frozen protocol drift')
    if control.verify_source(root) != manifest['source_manifest_sha256']:
        raise ValueError('Literature manifest drift')
    expected = {(b, s, m, a) for b in [*protocol.BENCHMARKS, 'billiger-de-50un']
                for s in (['test'] if b == 'billiger-de-50un' else ['prompt', 'route', 'test'])
                for m in protocol.MODELS for a in control.ARMS}
    cells = manifest['cells']
    keys = [(c['benchmark'], c['split'], c['model'], c['arm']) for c in cells]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Incomplete or duplicate control matrix')
    data = {}
    for c in cells:
        b, s, m, a = c['benchmark'], c['split'], c['model'], c['arm']
        if (b, s) not in data:
            cases = json.loads((target/'data'/b/(s+'.json')).read_text())
            if len({x['pair_id'] for x in cases}) != len(cases):
                raise ValueError('Duplicate pair')
            if any(x['benchmark'] != b or protocol.digest(x['records']) != x['input_hash'] for x in cases):
                raise ValueError('Input identity mismatch')
            data[b, s] = cases
        cases = data[b, s]
        if len(cases) != c['pairs'] or protocol.digest(cases) != c['input_data_sha256']:
            raise ValueError('Frozen dataset drift')
        h = hashlib.sha256()
        for case in cases:
            h.update(protocol.canonical(control.payload(case, m, a, protocol)).encode()+b'\n')
        if h.hexdigest() != c['ordered_payloads_sha256']:
            raise ValueError('Ordered request drift')
    if sum(c['pairs'] for c in cells) != manifest['calls_per_run_before_retries']:
        raise ValueError('Call inventory mismatch')
    return cells


def prepare(root, output):
    if output.exists():
        raise FileExistsError(output)
    _, protocol, orchestrate, billiger50 = load_modules(root)
    orchestrate.verify()
    billiger50.verify()
    manifest = json.loads((root/MANIFEST).read_text())
    cells = verify_matrix(root, root, manifest, protocol)
    for run_id in (2, 3):
        verify_replica(root, run_id)
    spec = {'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'prepared_not_admitted',
            'payload_manifest_sha256': sha(root/MANIFEST),
            'source_sha256': {p: sha(root/p) for p in SOURCES},
            'study_runs': [1, 2, 3], 'stage': 'peeters', 'namespace': NAMESPACE,
            'workers_per_model': WORKERS, 'models_parallel': len(protocol.MODELS),
            'order': ['prompt', 'route', 'test'], 'selection': 'none; both fixed controls',
            'cells_per_run': len(cells), 'calls_per_run_before_retries': sum(c['pairs'] for c in cells),
            'budget_cap_usd': 100, 'execution_admitted': False,
            'cost_quote_available': False, 'paid_calls_from_preparation': 0,
            'recovery': 'No automatic stage repeat. Terminal accounting and scoped provenance review required.'}
    with output.open('x') as f:
        f.write(json.dumps(spec, indent=2)+'\n')
    return spec


def validate_admission(admission, expected, budget):
    if not expected['slurm_job_id'] or any(admission.get(k) != v for k, v in expected.items()):
        raise ValueError('Peeters admission identity or hash mismatch')
    import math
    if any(type(budget[k]) not in (int, float) or not math.isfinite(budget[k]) or budget[k] < -1e-8
           for k in ['spent', 'reserved']):
        raise ValueError('Invalid shared budget')
    if budget['spent']+budget['reserved']+.05 > 100:
        raise ValueError('Shared project spending ceiling reached')


def validate_history(target, run_id):
    if (target/'artifacts/peeters_stage.jsonl').exists():
        raise ValueError('Peeters stage already attempted; reviewed recovery required')
    if any((target/'runs'/NAMESPACE).rglob('*.jsonl')):
        raise ValueError('Unadmitted existing Peeters responses')
    if run_id != 1:
        rows = [json.loads(x) for x in (target/'artifacts/stage_execution.jsonl').read_text().splitlines()]
        if any(x['run_id'] != run_id for x in rows):
            raise ValueError('Mixed replica journal')
        states = {x['stage']: x['status'] for x in rows}
        if states.get('core') != 'complete' or 'started' in states.values():
            raise ValueError('Core incomplete or unresolved replica writer')
    status = json.loads((target/'artifacts/progress.json').read_text())
    if status.get('smoke', {}).get('status') != 'complete':
        raise ValueError('Technical smoke not complete')


def install_adapter(run, protocol, root, run_id, manifest_sha):
    run.payload = lambda case, model, arm: control.payload(case, model, arm, protocol)
    run.account = shared_account(run.account, root)
    original = run.request

    def tagged(*args, **kwargs):
        row = original(*args, **kwargs)
        row.update(study_run_id=run_id, peeters_payload_manifest_sha256=manifest_sha)
        return row

    run.request = tagged


def run_matrix(run, cells):
    # Finish all models' development records before the first test cell.
    for split in ('prompt', 'route', 'test'):
        selected = [c for c in cells if c['split'] == split]
        def model_cells(model):
            for c in selected:
                if c['model'] == model:
                    run.run_cell(model, c['arm'], c['benchmark'], split,
                                 workers=WORKERS, namespace=NAMESPACE)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len({c['model'] for c in selected})) as pool:
            futures = [pool.submit(model_cells, m) for m in sorted({c['model'] for c in selected})]
            for future in concurrent.futures.as_completed(futures):
                future.result()


def execute(root, run_id, admission_path, policy_path):
    target = target_path(root, run_id)
    if target.resolve() != target:
        raise ValueError('Unexpected run directory symlink')
    if run_id != 1:
        verify_replica(root, run_id)
    spec = json.loads((root/EXECUTION).read_text())
    if sha(root/MANIFEST) != spec['payload_manifest_sha256']:
        raise ValueError('Payload manifest changed')
    for p, h in spec['source_sha256'].items():
        if sha(root/p) != h or (p.startswith('src/') and sha(target/p) != h):
            raise ValueError('Execution source drift: '+p)
    lockpath = target/'artifacts'/('peeters_writer.lock' if run_id == 1 else 'parallel_writer.lock')
    with (root/'artifacts/operations/repetition_writer.lock').open('a') as legacy, lockpath.open('a') as lock:
        fcntl.flock(legacy, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if run_id != 1 and (target/'execution_admission.json').exists():
            raise ValueError('Legacy replica admission exists')
        validate_history(target, run_id)
        job = os.environ.get('SLURM_JOB_ID')
        reviewed = review(root, policy_path, own_job=job)
        if not any(x.get('self') for x in reviewed['reviewed_jobs']):
            raise ValueError('Actual reviewed Slurm job absent')
        expected = {'slurm_job_id': job, 'run_id': run_id, 'stage': 'peeters',
                    'target': str(target), 'execution_manifest_sha256': sha(root/EXECUTION),
                    'parallel_policy_sha256': reviewed['policy_sha256'], 'budget_cap_usd': 100}
        admission = json.loads(admission_path.read_text())
        validate_admission(admission, expected, json.loads((root/'artifacts/budget.json').read_text()))
        run, protocol, orchestrate, billiger50 = load_modules(target)
        orchestrate.verify()
        billiger50.verify()
        cells = verify_matrix(root, target, json.loads((root/MANIFEST).read_text()), protocol)
        install_adapter(run, protocol, root, run_id, spec['payload_manifest_sha256'])
        with (target/'artifacts/peeters_stage.jsonl').open('x') as journal:
            def log(status):
                journal.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'status': status,
                    **expected, 'admission_sha256': sha(admission_path)})+'\n')
                journal.flush()
                os.fsync(journal.fileno())
            log('started')
            try:
                run_matrix(run, cells)
            except BaseException:
                log('failed')
                raise
            log('complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'run'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-id', type=int, choices=[1, 2, 3])
    p.add_argument('--admission', type=Path)
    p.add_argument('--parallel-policy', type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    if args.command == 'prepare':
        print(json.dumps(prepare(root, root/EXECUTION), indent=2))
    else:
        if args.run_id is None or args.admission is None or args.parallel_policy is None:
            p.error('run requires --run-id, --admission and --parallel-policy')
        execute(root, args.run_id, args.admission.resolve(), args.parallel_policy.resolve())
