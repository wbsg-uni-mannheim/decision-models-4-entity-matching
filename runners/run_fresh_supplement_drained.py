"""Single admitted fresh supplement stage; never submits or releases jobs.

The full historical supplement scenario is charged to planning headroom, without
assuming cache savings. Actual requests use the existing atomic USD200 binding.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types

from authorized_recovery_200 import APPROVAL, APPROVAL_SHA, PINNED, SELF as BINDING, authority, make_wrapper
from drained_inference import durable_append
import fresh_supplement_drained as component
from repeated_runs import replica_path, sha, verify
from resume_schema_once import terminal_steps

SELF = 'artifacts/operations/run_fresh_supplement_drained.py'
ATTEMPT = 'artifacts/supplement_drained_v1'
COST = 'artifacts/verification/fixed_extension_cost_scenarios_20261015.json'
COST_SHA = '0c23a167ff9dd794dfb1b1f23a5d8d13388f961399db1585929ca98fd806f2c3'
COST_PROOF = 'artifacts/verification/fixed_extension_cost_verification_20261015.json'
COST_PROOF_SHA = '89330a3bddb26adf9808037a7b1f33a8c0b2bf1e83628cdf1e308f5cd44d02a6'
REQUIRED = {SELF, BINDING, APPROVAL, 'artifacts/operations/fresh_supplement_drained.py',
            'artifacts/operations/diagnose_fresh_baseline.py', 'artifacts/operations/operational_journal.py',
            'artifacts/operations/repeated_runs.py', 'artifacts/operations/resume_schema_once.py'} | {
                f'artifacts/operations/{name}.py' for name in PINNED}


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines()]


def bind(fn, **updates):
    return types.FunctionType(fn.__code__, {**fn.__globals__, **updates}, fn.__name__, fn.__defaults__, fn.__closure__)


def validate(spec, own, run_id, target):
    expected = {'admitted': True, 'stage': 'supplement', 'run_id': run_id,
                'slurm_job_id': own, 'replica': str(target), 'budget_cap_usd': 200,
                'budget_authority_sha256': APPROVAL_SHA, 'max_invocations': 1,
                'historical_unknown_outcomes_disclosed': True}
    if not own or type(run_id) is not int or run_id not in (2, 3) or any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('Actual reviewed supplement admission required')
    if not REQUIRED <= spec.get('source_sha256', {}).keys():
        raise ValueError('Executable provenance incomplete')


def cache_inventory(target, protocol):
    expected = {f'runs/quality/{m}/B0/{component.BENCHMARK}/test{s}' for m in protocol.MODELS
                for s in ('.jsonl', '.failures.jsonl', '.executions.jsonl', '.operations.jsonl')}
    observed = {str(p.relative_to(target)) for p in (target/'runs/quality').glob('*/B0/'+component.BENCHMARK+'/*.jsonl')}
    if not observed <= expected:
        raise ValueError('Unreviewed supplement journal')
    return {p: sha(target/p) if (target/p).exists() else None for p in sorted(expected)}


def core_history(history, run_id, core_job):
    if any(r.get('run_id') != run_id or r.get('stage') not in ('smoke', 'core') for r in history):
        raise ValueError('Mixed or later replica stage history')
    smoke = [r for r in history if r['stage'] == 'smoke']
    core = [r for r in history if r['stage'] == 'core']
    if [r['status'] for r in smoke] != ['started', 'complete']:
        raise ValueError('Smoke prerequisite incomplete')
    if [r['status'] for r in core] not in (['started', 'complete'], ['started', 'failed', 'started', 'complete']):
        raise ValueError('Core prerequisite incomplete or unreviewed recovery history')
    if any(str(r['slurm_job_id']) != core_job for r in core[-2:]):
        raise ValueError('Core completion belongs to a different job')


def successful_core(accounting, job):
    terminal_steps(accounting, job)
    parsed = [s.split('|') for s in accounting.splitlines() if s.strip()]
    if any(r[1] != 'COMPLETED' or r[2] != '0:0' for r in parsed):
        raise ValueError('Every core step must have completed successfully')


def planning(root, budget, protocol):
    if sha(root/COST) != COST_SHA or sha(root/COST_PROOF) != COST_PROOF_SHA:
        raise ValueError('Verified fixed extension cost sources changed')
    cost, proof = read(root/COST), read(root/COST_PROOF)
    if proof['scenario_sha256'] != COST_SHA or cost['quote_or_hard_bound'] is not False:
        raise ValueError('Historical planning provenance invalid')
    # Only supplement source journals are relevant here. Other cost-plan replica
    # absences legitimately change while independently admitted cores run.
    source = cache_inventory(root, protocol)
    if any(cost['source_journal_sha256_or_absent'].get(p, 'missing') != h for p, h in source.items()):
        raise ValueError('Supplement cost source journal changed')
    stage = cost['stages']['supplement']
    if stage != proof['stages']['supplement'] or stage['cells'] != 7 or stage['prospective_responses_per_run'] != 31059:
        raise ValueError('Wrong supplement cost matrix')
    values = [budget.get('spent'), budget.get('reserved'), stage['known_proxy_usd'], stage['historical_unknown_scenario_usd']]
    if any(type(v) not in (float, int) or not math.isfinite(v) or v < -1e-8 for v in values):
        raise ValueError('Invalid budget/planning value')
    if math.fsum(values) > 200:
        raise ValueError('Full supplement planning scenario exceeds current headroom')
    return {'scenario_sha256': COST_SHA, 'verification_sha256': COST_PROOF_SHA,
            'full_stage_scenario': stage, 'quote_or_hard_bound': False, 'future_cache_discount': False}


def verify_ready(root, target, run_id, spec, own, replica, protocol):
    validate(spec, own, run_id, target)
    authority(root)
    if spec.get('runner_sha256') != replica['runner_sha256']:
        raise ValueError('Replica provenance changed')
    for p, digest in spec['source_sha256'].items():
        if sha(root/p) != digest:
            raise ValueError('Operational source drift: '+p)
    journal = target/'artifacts/stage_execution.jsonl'
    core_job = str(spec['core_job_id'])
    core_history(rows(journal), run_id, core_job)
    if sha(journal) != spec['core_stage_journal_sha256'] or sha(target/'artifacts/progress.json') != spec['progress_sha256']:
        raise ValueError('Core stage/progress changed since admission')
    progress = read(target/'artifacts/progress.json')
    required = {'smoke': 'complete', 'AP1_baseline': 'complete', 'AP2_development': 'complete',
                'AP2_confirmation': 'complete', 'AP1_timing': 'complete', 'AP3_AP4_analysis': 'complete_automated'}
    if set(progress) != set(required) or any(progress[k]['status'] != v for k, v in required.items()):
        raise ValueError('Every core phase must be complete')
    if cache_inventory(target, protocol) != spec['cache_sha256_or_absent']:
        raise ValueError('Supplement cache changed since admission')
    audited = component.inventory(target, run_id, protocol)
    if any(c['failure_rows'] or c['operations']['events'] for c in audited):
        raise ValueError('Existing supplement failure requires separate terminal review')
    budget = read(root/'artifacts/budget.json')
    cost = planning(root, budget, protocol)
    accounting = subprocess.check_output(['sacct', '-n', '-j', core_job,
        '--format=JobID,State,ExitCode,End', '-P'], text=True, env={**os.environ, 'TZ': 'UTC'})
    successful_core(accounting, core_job)
    policy = make_wrapper(root, 'core').review(root, root/spec['parallel_policy'], own)
    if policy['policy_sha256'] != spec['parallel_policy_sha256'] or not any(j.get('self') for j in policy['reviewed_jobs']):
        raise ValueError('Current policy does not admit this job')
    dependencies = {}
    for item in policy['reviewed_jobs']:
        if item.get('scope') == 'different project':
            continue
        job = item['id']
        meta = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
        dep = next((v.split('=', 1)[1] for v in meta.split() if v.startswith('Dependency=')), '')
        if '?' in dep or (dep not in ('(null)', 'null', '') and any(not d.startswith('afterok:') for d in dep.split(','))):
            raise ValueError('Only conjunctive afterok joins are accepted')
        dependencies[job] = dep
        if not item.get('self'):
            script = subprocess.check_output(['scontrol', 'write', 'batch_script', job, '-'], text=True)
            if str(target) in script or f'--run-id {run_id}' in script or f'--run-id={run_id}' in script:
                raise ValueError('Another queued or running writer targets this replica')
    if dependencies != spec['reviewed_dependencies']:
        raise ValueError('All queue joins must match the current admission')
    return {'budget': budget, 'cost': cost, 'policy': policy, 'dependencies': dependencies,
            'core_accounting_utc': accounting, 'audited_cells': audited}


def finish_report(target, run_id, supplement, analysis, orchestration, protocol):
    checked = component.inventory(target, run_id, protocol)
    if not all(c['complete'] for c in checked):
        raise ValueError('Full raw supplement required before report')
    loader = bind(analysis.load, PROJECT=target)
    report = bind(supplement.analyze, PROJECT=target, load=loader, verify=supplement.verify)
    report()
    stamp = bind(orchestration.stamp, PROJECT=target)
    stamp('supplement_billiger50', 'complete_automated', '7 cells and historical context complete; manuscript integration pending')
    return {str(p.relative_to(target)): sha(p) for p in sorted((target/'artifacts/results/billiger50').iterdir()) if p.is_file()}


def execute_bound(root, target, run_id, admission_path, spec, own, replica, inference, supplement, protocol, analysis, orchestration):
    validate(spec, own, run_id, target)
    with (root/'artifacts/operations/repetition_writer.lock').open('a') as shared, \
            (target/'artifacts/parallel_writer.lock').open('a') as exclusive:
        fcntl.flock(shared, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = target/ATTEMPT
        if attempt.exists() or (target/'artifacts/results/billiger50').exists():
            raise ValueError('Previous supplement attempt/report requires separate review')
        supplement.verify()
        checked = verify_ready(root, target, run_id, spec, own, replica, protocol)
        admission_sha = sha(admission_path)
        prior_sha = sha(target/'artifacts/stage_execution.jsonl')
        attempt.mkdir(exist_ok=False)
        def log(status, **extra):
            record = {'utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
                      'stage': 'supplement', 'status': status, 'slurm_job_id': own,
                      'core_job_id': spec['core_job_id'], 'prior_stage_journal_sha256': prior_sha,
                      'admission_sha256': admission_sha, 'launcher_sha256': sha(root/SELF),
                      'parallel_policy_sha256': spec['parallel_policy_sha256'],
                      'budget_authority_sha256': APPROVAL_SHA, 'historical_unjournaled_outcomes_resolved': False, **extra}
            durable_append(attempt/'execution.jsonl', record, protocol.canonical)
            durable_append(target/'artifacts/stage_execution.jsonl', record, protocol.canonical)
        log('started', verification=checked)
        try:
            result = component.run_supplement(root, target, run_id, inference, supplement, protocol, attempt/'inference')
            if not result['inference_complete']:
                raise RuntimeError('Supplement failed after all model tasks drained')
            verify(root, run_id)
            if sha(admission_path) != admission_sha:
                raise ValueError('Admission changed during execution')
            report = finish_report(target, run_id, supplement, analysis, orchestration, protocol)
            log('complete', component=result, report_sha256=report)
        except BaseException as exc:
            bind(orchestration.stamp, PROJECT=target)('supplement_billiger50', 'needs_attention', type(exc).__name__)
            log('failed', exception_type=type(exc).__name__)
            raise
    return 0


def execute(root, run_id, admission_path):
    root = root.resolve(); target = replica_path(root, run_id)
    spec = read(admission_path); own = os.environ.get('SLURM_JOB_ID')
    validate(spec, own, run_id, target)
    if any(m in sys.modules for m in ('run', 'protocol', 'billiger50', 'analyze', 'orchestrate')):
        raise ValueError('Fresh Python process required')
    replica = verify(root, run_id)
    if target.resolve() != target or Path(replica['replica']) != target:
        raise ValueError('Wrong replica path')
    sys.path.insert(0, str(target/'src'))
    import run, billiger50, protocol, analyze, orchestrate
    if any(m.PROJECT != target for m in (run, billiger50, analyze, orchestrate)):
        raise ValueError('Mixed replica modules')
    orchestrate.verify()
    return execute_bound(root, target, run_id, admission_path, spec, own, replica, run, billiger50, protocol, analyze, orchestrate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', type=int, choices=(2, 3), required=True)
    parser.add_argument('--admission', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.root, args.run_id, args.admission.resolve()))
