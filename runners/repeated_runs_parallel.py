"""Separate parallel launcher; preserves the original frozen replica runner.

One writer per replica, shared atomic budget accounting across all replicas.
Admission records and Slurm policy entries must be prepared for actual jobs.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import runpy
import sys

from parallel_admission import review
from repeated_runs import STAGES as FROZEN_STAGES, replica_path, sha, shared_account, verify

# This additional stage does not change the frozen runner or replica source tree.
STAGES = {**FROZEN_STAGES, 'baseline': None}

PREREQUISITES = {
    'smoke': [], 'core': ['smoke'], 'supplement': ['core'],
    'screen': ['core'], 'calibration': ['core'],
    'confirmation': ['screen', 'calibration', 'supplement'],
    'stability': ['screen'], 'cascades': ['core', 'supplement'],
    'live': ['cascades', 'stability'], 'variants': ['core'],
    'schema': ['core', 'supplement'], 'rules': ['confirmation', 'schema'],
    'errors': ['core', 'supplement'],
    'baseline': ['smoke'],
}


def validate_admission(admission, run_id, stage, job, spec, launcher_sha, policy_sha, budget):
    if not job or admission.get('slurm_job_id') != job:
        raise ValueError('Actual reviewed Slurm job required')
    expected = {'run_id': run_id, 'stage': stage, 'budget_cap_usd': 100,
                'runner_sha256': spec['runner_sha256'], 'launcher_sha256': launcher_sha,
                'parallel_policy_sha256': policy_sha, 'replica': spec['replica']}
    if any(admission.get(k) != v for k, v in expected.items()):
        raise ValueError('Parallel stage admission identity or hash mismatch')
    if any(type(budget[k]) not in (int, float) or not math.isfinite(budget[k]) or budget[k] < -1e-8
           for k in ['spent', 'reserved']):
        raise ValueError('Invalid budget snapshot')
    if budget['spent'] + budget['reserved'] + .05 > 100:
        raise ValueError('Shared project budget exhausted')
    # This is only a preflight; frozen account() reserves atomically per call.


def validate_history(rows, run_id, stage):
    if any(r.get('run_id') != run_id for r in rows):
        raise ValueError('Mixed study run journal')
    if any(r.get('stage') == stage for r in rows):
        raise ValueError('Stage already attempted; reviewed recovery required')
    overlaps = {'baseline': {'core', 'supplement'},
                'core': {'baseline'}, 'supplement': {'baseline'}}
    if any(r.get('stage') in overlaps.get(stage, set()) for r in rows):
        raise ValueError('Overlapping baseline stage already attempted; reviewed continuation required')
    for prerequisite in PREREQUISITES[stage]:
        states = [r['status'] for r in rows if r.get('stage') == prerequisite]
        if not states or states[-1] != 'complete':
            raise ValueError('Unfinished prerequisite: '+prerequisite)
    for other in {r['stage'] for r in rows}:
        states = [r['status'] for r in rows if r['stage'] == other]
        if states[-1] == 'started':
            raise ValueError('Unresolved previous writer; inspect terminal Slurm steps')


def baseline_only(target):
    """B0 quality matrix only: original development splits, test and billiger50.

    No AP2, selection, timing, analysis subprocess or other experiment is invoked.
    Smoke answers in this same replica remain reusable as in the frozen pipeline.
    """
    import billiger50
    import orchestrate
    import protocol
    import run
    if any(module.PROJECT != target for module in [run, orchestrate, billiger50]):
        raise RuntimeError('Wrong baseline module workspace')
    # Verify every required dataset before the first new request.
    orchestrate.verify()
    billiger50.verify()
    status = json.loads((target/'artifacts/progress.json').read_text())
    if status.get('smoke', {}).get('status') != 'complete':
        raise ValueError('Baseline requires completed technical smoke test')
    stage = 'baseline_only'
    orchestrate.stamp(stage, 'running')
    try:
        orchestrate.parallel(list(protocol.MODELS), ['B0'], ['prompt', 'route', 'test'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(protocol.MODELS)) as pool:
            futures = [pool.submit(run.run_cell, model, 'B0', billiger50.BENCHMARK,
                                   'test', workers=16) for model in protocol.MODELS]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    except Exception as exc:
        orchestrate.stamp(stage, 'needs_attention', str(exc))
        raise
    orchestrate.stamp(stage, 'complete', 'B0 quality only; no AP2, timing or cascade execution')


def execute(root, run_id, stage, admission_path, policy_path):
    root = root.resolve()
    spec = verify(root, run_id)
    launcher_sha = sha(Path(__file__).resolve())
    target = replica_path(root, run_id)
    if Path(spec['replica']) != target or target.resolve() != target:
        raise ValueError('Replica path identity mismatch')
    with (root/'artifacts/operations/repetition_writer.lock').open('a') as legacy_lock, \
            (target/'artifacts/parallel_writer.lock').open('a') as lock:
        # Shared among parallel launchers, exclusive in the frozen old launcher.
        # Thus legacy execution cannot race the per-replica admission check.
        fcntl.flock(legacy_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Refuse use alongside the original serial launcher in this replica.
        legacy = target/'execution_admission.json'
        if legacy.exists():
            raise ValueError('Legacy execution admission exists; reconcile before parallel launch')
        job = os.environ.get('SLURM_JOB_ID')
        reviewed = review(root, policy_path, own_job=job)
        if not any(r.get('self') for r in reviewed['reviewed_jobs']):
            raise ValueError('This job is absent from reviewed project policy')
        admission = json.loads(admission_path.read_text())
        validate_admission(admission, run_id, stage, job, spec, launcher_sha,
                           reviewed['policy_sha256'], json.loads((root/'artifacts/budget.json').read_text()))
        journal = target/'artifacts/stage_execution.jsonl'
        rows = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        validate_history(rows, run_id, stage)
        if any(m in sys.modules for m in ['run', 'protocol', 'orchestrate', 'analyze']):
            raise RuntimeError('Fresh Python process required')
        sys.path.insert(0, str(target/'src'))
        import run
        if run.PROJECT != target:
            raise RuntimeError('Wrong inference workspace')
        run.account = shared_account(run.account, root)
        original_request = run.request

        def tagged_request(*args, **kwargs):
            row = original_request(*args, **kwargs)
            row['study_run_id'] = run_id
            return row

        run.request = tagged_request
        def log(status):
            with journal.open('a') as handle:
                handle.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
                    'run_id': run_id, 'stage': stage, 'status': status, 'slurm_job_id': job,
                    'admission_sha256': sha(admission_path), 'launcher_sha256': launcher_sha,
                    'parallel_policy_sha256': reviewed['policy_sha256']})+'\n')
                handle.flush()
                os.fsync(handle.fileno())
        command = STAGES[stage]
        log('started')
        os.chdir(target)
        try:
            if stage == 'baseline':
                baseline_only(target)
            else:
                sys.argv = [str(target/'src'/command[0]), *command[1:]]
                runpy.run_path(sys.argv[0], run_name='__main__')
        except BaseException:
            log('failed')
            raise
        log('complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-id', type=int, choices=[2, 3], required=True)
    p.add_argument('--stage', choices=STAGES, required=True)
    p.add_argument('--admission', type=Path, required=True)
    p.add_argument('--parallel-policy', type=Path, required=True)
    args = p.parse_args()
    execute(args.root, args.run_id, args.stage, args.admission.resolve(), args.parallel_policy.resolve())
