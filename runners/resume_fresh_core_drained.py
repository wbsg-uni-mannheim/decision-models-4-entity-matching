"""Bounded core continuation; no submission, admission creation or budget override.

Supports only a diagnosed initial core failure during B0. Preserves the old
stage journal and appends recovery start/end with immutable prior provenance.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from drained_inference import durable_append
from fresh_core_drained import run_phases
from operational_journal import inspect_operations
from parallel_admission import review
from repeated_runs import replica_path, sha, verify
from resume_schema_once import terminal_steps

REQUIRED = {'artifacts/operations/resume_fresh_core_drained.py',
            'artifacts/operations/fresh_core_drained.py', 'artifacts/operations/drained_inference.py',
            'artifacts/operations/operational_journal.py', 'artifacts/operations/parallel_admission.py',
            'artifacts/operations/repeated_runs.py', 'artifacts/operations/resume_schema_once.py'}
PHASES = {'AP1_baseline', 'AP2_development', 'AP2_confirmation', 'AP1_timing'}
ATTEMPT = 'artifacts/core_drained_recovery_v1'
APPROVAL = 'artifacts/operations/approval_runs23_aaron_20261014.json'
APPROVAL_SHA = '2e5703ca1fddc3f788957e4b87a2414f1608a16b1a0c12d48f50df366444c849'


def read(path):
    return json.loads(path.read_text())


def history_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate_history(rows, run_id, predecessor):
    if any(r.get('run_id') != run_id or r.get('stage') not in ('smoke', 'core') for r in rows):
        raise ValueError('Mixed or later stage history')
    smoke = [r for r in rows if r['stage'] == 'smoke']
    core = [r for r in rows if r['stage'] == 'core']
    if ([r['status'] for r in smoke] != ['started', 'complete']
            or [r['status'] for r in core] != ['started', 'failed']
            or any(r['slurm_job_id'] != predecessor for r in core)):
        raise ValueError('Only one diagnosed failed initial core is supported')


def validate_cost(cost, diagnosis_sha, budget):
    if cost.get('diagnosis_sha256') != diagnosis_sha or set(cost.get('phases', {})) != PHASES:
        raise ValueError('Complete diagnosis-linked phase cost plan required')
    values = [cost['phases'][p].get(k) for p in PHASES
              for k in ('known_proxy_usd', 'historical_unknown_scenario_usd')]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Invalid planning values')
    if cost.get('source_verification_sha256') is None or cost.get('quote_or_hard_bound') is not False:
        raise ValueError('Verified historical planning provenance required')
    if any(type(budget.get(k)) not in (int, float) or not math.isfinite(budget[k]) or budget[k] < -1e-8
           for k in ('spent', 'reserved')):
        raise ValueError('Invalid budget')
    if budget['spent']+budget['reserved']+max(.05, math.fsum(values)) > 100:
        raise ValueError('Full core planning scenario exceeds current headroom')


def verify_ready(root, target, run_id, spec, own, replica):
    expected = {'admitted': True, 'run_id': run_id, 'stage': 'core', 'slurm_job_id': own,
                'budget_cap_usd': 100, 'max_invocations': 1, 'replica': str(target),
                'historical_unknown_outcomes_disclosed': True, 'runner_sha256': replica['runner_sha256']}
    if not own or type(run_id) is not int or run_id not in (2, 3) or any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('Actual reviewed fresh-core recovery admission required')
    if sha(root/APPROVAL) != APPROVAL_SHA:
        raise ValueError('Original Run2/3 authority changed')
    if not REQUIRED <= spec.get('source_sha256', {}).keys():
        raise ValueError('Executable provenance incomplete')
    for rel, expected_sha in spec['source_sha256'].items():
        if sha(root/rel) != expected_sha:
            raise ValueError('Operational source drift: '+rel)
    diagnosis = read(root/spec['diagnosis'])
    if sha(root/spec['diagnosis']) != spec['diagnosis_sha256']:
        raise ValueError('Diagnosis changed')
    predecessor = spec['predecessor']
    if (diagnosis['run_id'] != run_id or diagnosis['predecessor'] != predecessor
            or diagnosis['status'] != 'terminal_partial_baseline_audited_not_recovered'):
        raise ValueError('Wrong initial core diagnosis')
    journal = target/'artifacts/stage_execution.jsonl'
    validate_history(history_rows(journal), run_id, predecessor)
    if sha(journal) != diagnosis['stage_journal_sha256'] or read(target/'artifacts/progress.json') != diagnosis['progress']:
        raise ValueError('Historical stage or progress changed')
    inventory = {str(p.relative_to(target)): sha(p) for p in (target/'runs').rglob('*.jsonl')}
    if inventory != diagnosis['response_files_sha256']:
        raise ValueError('Audited replica cache inventory changed')
    cost = read(root/spec['cost_plan'])
    if sha(root/spec['cost_plan']) != spec['cost_plan_sha256']:
        raise ValueError('Cost plan changed')
    if sha(root/spec['cost_source_verification']) != cost.get('source_verification_sha256'):
        raise ValueError('Cost source verification changed')
    budget = read(root/'artifacts/budget.json')
    validate_cost(cost, spec['diagnosis_sha256'], budget)
    accounting = subprocess.check_output(['sacct', '-n', '-j', predecessor,
        '--format=JobID,State,ExitCode,End', '-P'], text=True, env={**os.environ, 'TZ': 'UTC'})
    terminal_steps(accounting, predecessor)
    policy = review(root, root/spec['parallel_policy'], own)
    if policy['policy_sha256'] != spec['parallel_policy_sha256'] or not any(r.get('self') for r in policy['reviewed_jobs']):
        raise ValueError('Current policy does not admit this job')
    checked_joins = {}
    for item in policy['reviewed_jobs']:
        if item.get('scope') == 'different project':
            continue
        job = item['id']
        meta = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
        dep = next((s.split('=', 1)[1] for s in meta.split() if s.startswith('Dependency=')), '')
        if predecessor in re.findall(r'\d+', dep):
            raise ValueError('A downstream dependency still references the failed predecessor')
        checked_joins[job] = dep
        if item.get('self'):
            continue
        script = subprocess.check_output(['scontrol', 'write', 'batch_script', job, '-'], text=True)
        same_run = (str(target) in script or f'--run-id {run_id}' in script
                    or f'--run-id={run_id}' in script)
        if same_run:
            raise ValueError('Another queued or running writer targets this replica')
    if checked_joins != spec.get('reviewed_dependencies'):
        raise ValueError('Queue dependencies differ from reviewed admission')
    return {'budget': budget, 'policy': policy, 'accounting_utc': accounting,
            'dependencies': checked_joins, 'planning_is_quote_or_hard_bound': False}


def execute(root, run_id, admission_path):
    root = root.resolve()
    if any(m in sys.modules for m in ('run', 'protocol', 'orchestrate', 'analyze')):
        raise ValueError('Fresh Python process required')
    target = replica_path(root, run_id)
    replica = verify(root, run_id)
    if target.resolve() != target or Path(replica['replica']) != target:
        raise ValueError('Wrong replica path')
    with (root/'artifacts/operations/repetition_writer.lock').open('a') as shared, \
            (target/'artifacts/parallel_writer.lock').open('a') as exclusive:
        fcntl.flock(shared, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = target/ATTEMPT
        if attempt.exists() or (target/'execution_admission.json').exists():
            raise ValueError('Prior recovery or legacy admission requires new diagnosis')
        spec = read(admission_path)
        own = os.environ.get('SLURM_JOB_ID')
        checked = verify_ready(root, target, run_id, spec, own, replica)
        sys.path.insert(0, str(target/'src'))
        import run
        import protocol
        import orchestrate
        if run.PROJECT != target or orchestrate.PROJECT != target:
            raise ValueError('Mixed replica modules')
        orchestrate.verify()
        attempt.mkdir(exist_ok=False)
        journal = target/'artifacts/stage_execution.jsonl'
        prior_sha = sha(journal)
        def log(status, **extra):
            record = {'utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
                      'stage': 'core', 'status': status, 'slurm_job_id': own,
                      'recovery': 'drained_v1', 'predecessor': spec['predecessor'],
                      'prior_stage_journal_sha256': prior_sha, 'admission_sha256': sha(admission_path),
                      'launcher_sha256': sha(Path(__file__)), 'parallel_policy_sha256': spec['parallel_policy_sha256'],
                      'historical_unjournaled_outcomes_resolved': False, **extra}
            durable_append(attempt/'execution.jsonl', record, protocol.canonical)
            durable_append(journal, record, protocol.canonical)
        log('started', verification=checked)
        try:
            result = run_phases(root, target, run_id, run, orchestrate, protocol, attempt/'phases')
            operations = []
            for path in sorted((target/'runs').rglob('*.operations.jsonl')):
                ns, model, variant, benchmark, name = path.relative_to(target/'runs').parts
                split = name.split('.')[0]
                cases = read(target/'data'/benchmark/(split+'.json'))
                if ns.startswith('timing_'):
                    cases = cases[:20]
                failure = path.with_name(split+'.failures.jsonl')
                failures = history_rows(failure) if failure.exists() else []
                operations.append(inspect_operations(path, cases, failures, protocol, model, variant, run_id))
            if any(r['unknown_outcome_events'] for r in operations):
                raise ValueError('Unknown operational outcome requires diagnosis')
            verify(root, run_id)
        except BaseException as exc:
            log('failed', exception_type=type(exc).__name__)
            raise
        log('complete', component=result, operations=operations,
            cost_scope='Journaled costs only; historical unjournaled outcomes remain unresolved')
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', type=int, choices=(2, 3), required=True)
    parser.add_argument('--admission', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.root, args.run_id, args.admission.resolve()))
