"""Isolated copies of frozen pipelines; all paid requests share the original ledger.

No paid execution is enabled by prepare. A reviewed Slurm admission record is
required before stage execution. Original source files and run1 remain unchanged.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import statistics
import sys
import types

STAGES = {
    'smoke': ('orchestrate.py', 'smoke'),
    'core': ('orchestrate.py', 'all'),
    'supplement': ('billiger50.py', 'run'),
    'screen': ('expanded_screen.py', 'run'),
    'calibration': ('expanded_calibration.py', 'run'),
    'confirmation': ('expanded_confirmation.py', 'run'),
    'stability': ('expanded_stability.py', 'run'),
    'cascades': ('expanded_cascades.py', 'analyze'),
    'live': ('expanded_cascades.py', 'live'),
    'variants': ('expanded_variants.py', 'run'),
    'schema': ('expanded_schema.py', 'run'),
    'rules': ('expanded_rules.py', 'run'),
    'errors': ('expanded_errors.py', 'analyze'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def replica_path(root, run_id):
    if run_id not in (2, 3):
        raise ValueError('Only fresh runs 2 and 3; run1 is never copied or reset')
    return root / 'replications' / f'run{run_id:02d}'


def prepare(root, run_id):
    root = root.resolve()
    target = replica_path(root, run_id)
    target.mkdir(parents=True, exist_ok=False)
    for name in ['src', 'configs', 'artifacts', 'runs', 'logs', 'manuscript']:
        (target/name).mkdir()
    frozen = {}
    for source in sorted((root/'src').glob('*.py')):
        shutil.copy2(source, target/'src'/source.name)
        frozen[str(source.relative_to(root))] = sha(source)
    # Explicit immutable inputs only. Never import run1's fitted selections/status.
    inputs = sorted({*(root/'configs').glob('*manifest.json'),
                     root/'configs/learned_rules.json'})
    for source in inputs:
        shutil.copy2(source, target/'configs'/source.name)
        frozen[str(source.relative_to(root))] = sha(source)
    (target/'data').symlink_to(root/'data', target_is_directory=True)
    (target/'.secrets').symlink_to(root/'.secrets', target_is_directory=True)
    (target/'artifacts/astra_agent').mkdir()
    acceptance = Path('artifacts/astra_agent/acceptance.json')
    shutil.copy2(root/acceptance, target/acceptance)
    frozen[str(acceptance)] = sha(root/acceptance)
    # Read-only views for forecasts. The patched account function writes ROOT,
    # never replaces this budget symlink with a private, independent ledger.
    (target/'artifacts/budget.json').symlink_to(root/'artifacts/budget.json')
    save(target/'artifacts/progress.json', {})
    record = {'run_id': run_id, 'root': str(root), 'replica': str(target),
              'created_at': datetime.now(timezone.utc).isoformat(),
              'immutable_sha256': frozen, 'runner_sha256': sha(Path(__file__)),
              'fresh_responses': True, 'reuse_across_runs': False,
              'shared_budget': str(root/'artifacts/budget.json'),
              'stage_commands': STAGES, 'execution_enabled': False,
              'selection_policy': 'Recompute each development selection/fit in this replica before its test stage',
              'scope': 'Original seven-model pipeline; new ensemble/Astra modules require separate manifests'}
    save(target/'replica.json', record)
    return record


def verify(root, run_id):
    root = root.resolve()
    target = replica_path(root, run_id)
    spec = json.loads((target/'replica.json').read_text())
    assert spec['run_id'] == run_id and spec['root'] == str(root)
    assert spec['runner_sha256'] == sha(Path(__file__)), 'Runner changed after preparation'
    for filename, expected in spec['immutable_sha256'].items():
        assert sha(target/filename) == expected, filename
        assert sha(root/filename) == expected, 'Original frozen source/input drift: ' + filename
    assert (target/'data').resolve() == root/'data'
    assert (target/'artifacts/budget.json').is_symlink()
    assert (target/'artifacts/budget.json').resolve() == root/'artifacts/budget.json'
    assert not (target/'runs').is_symlink()
    return spec


def shared_account(account_function, root):
    """Clone only account's globals; run.request keeps its replica PROJECT."""
    scope = dict(account_function.__globals__)
    scope['PROJECT'] = root
    return types.FunctionType(account_function.__code__, scope,
                              account_function.__name__, account_function.__defaults__,
                              account_function.__closure__)


def execute(root, run_id, stage):
    root = root.resolve()
    spec = verify(root, run_id)
    target = replica_path(root, run_id)
    admission = json.loads((target/'execution_admission.json').read_text())
    assert admission['run_id'] == run_id and admission['stage'] == stage
    assert admission['slurm_job_id'] == os.environ.get('SLURM_JOB_ID')
    assert admission['all_other_inference_writers_ended'] is True
    assert admission['runner_sha256'] == spec['runner_sha256']
    assert admission['budget_cap_usd'] == 100
    with (root/'artifacts/operations/repetition_writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = json.loads((root/'artifacts/budget.json').read_text())
        assert abs(budget['reserved']) < 1e-8 and budget['spent'] + .05 <= 100
        sys.path.insert(0, str(target/'src'))
        if 'run' in sys.modules:
            raise RuntimeError('Execute in a fresh process to prevent mixed run modules')
        import run
        assert run.PROJECT == target
        run.account = shared_account(run.account, root)
        original_request = run.request

        def tagged_request(*args, **kwargs):
            row = original_request(*args, **kwargs)
            row['study_run_id'] = run_id
            return row

        run.request = tagged_request
        # All frozen paid modules call run.account. The only original subprocess
        # is core analyze.py, which reads replica caches and makes no paid calls.
        command = STAGES[stage]
        journal = target/'artifacts/stage_execution.jsonl'

        def log(status):
            with journal.open('a') as f:
                f.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
                                    'run_id': run_id, 'stage': stage, 'status': status,
                                    'slurm_job_id': os.environ['SLURM_JOB_ID']})+'\n')
                f.flush()
                os.fsync(f.fileno())

        log('started')
        sys.argv = [str(target/'src'/command[0]), *command[1:]]
        os.chdir(target)
        try:
            runpy.run_path(sys.argv[0], run_name='__main__')
        except BaseException:
            log('failed')
            raise
        log('complete')


def three_run_summary(values):
    """No SD from partial runs, duplicated run IDs, NaN or zero-filled missingness."""
    if set(values) != {1, 2, 3}:
        return {'status': 'incomplete', 'observed_runs': sorted(values), 'mean': None, 'sd': None}
    nums = list(values.values())
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in nums):
        raise ValueError('Finite actual values required')
    return {'status': 'complete', 'n': 3, 'mean': statistics.mean(nums),
            'sd': statistics.stdev(nums), 'ddof': 1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'verify', 'execute'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-id', type=int, required=True, choices=[2, 3])
    p.add_argument('--stage', choices=STAGES)
    args = p.parse_args()
    if args.mode == 'execute':
        assert args.stage
        execute(args.root, args.run_id, args.stage)
    else:
        print(json.dumps((prepare if args.mode == 'prepare' else verify)(args.root, args.run_id), indent=2))


if __name__ == '__main__':
    main()
