"""Single admitted fresh calibration stage; never submits or releases jobs.

The full historical calibration scenario is charged to planning headroom, without
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
import fresh_calibration_drained as component
import fresh_screen_drained as screen_component
from diagnose_fresh_baseline import audit_rows
from operational_journal import inspect_operations
import fresh_supplement_drained as supplement_component
from run_fresh_screen_drained import prerequisite_history as screen_prerequisite_history
from repeated_runs import replica_path, sha, verify
from resume_schema_once import terminal_steps

SELF = 'artifacts/operations/run_fresh_calibration_drained.py'
ATTEMPT = 'artifacts/calibration_drained_v1'
COST = 'artifacts/verification/fixed_extension_cost_scenarios_20261015.json'
COST_SHA = '0c23a167ff9dd794dfb1b1f23a5d8d13388f961399db1585929ca98fd806f2c3'
COST_PROOF = 'artifacts/verification/fixed_extension_cost_verification_20261015.json'
COST_PROOF_SHA = '89330a3bddb26adf9808037a7b1f33a8c0b2bf1e83628cdf1e308f5cd44d02a6'
REQUIRED = {'artifacts/operations/run_fresh_supplement_drained.py',
            'artifacts/operations/fresh_supplement_drained.py', SELF, BINDING, APPROVAL, 'artifacts/operations/fresh_screen_drained.py', 'artifacts/operations/fresh_calibration_drained.py',
            'artifacts/operations/run_fresh_screen_drained.py',
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
    expected = {'admitted': True, 'stage': 'calibration', 'run_id': run_id,
                'slurm_job_id': own, 'replica': str(target), 'budget_cap_usd': 200,
                'budget_authority_sha256': APPROVAL_SHA, 'max_invocations': 1,
                'historical_unknown_outcomes_disclosed': True}
    if not own or type(run_id) is not int or run_id not in (2, 3) or any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('Actual reviewed calibration admission required')
    if not REQUIRED <= spec.get('source_sha256', {}).keys():
        raise ValueError('Executable provenance incomplete')


def cache_inventory(target, calibration, splits=('calibration_extra', 'route', 'test')):
    expected = {f'runs/quality/{m}/B0/{b}/{split}{suffix}'
                for m in calibration.DECISION_MODELS for b in calibration.BENCHMARKS
                for split in splits for suffix in ('.jsonl', '.failures.jsonl', '.executions.jsonl', '.operations.jsonl')}
    observed = {str(p.relative_to(target)) for m in calibration.DECISION_MODELS
                for b in calibration.BENCHMARKS for split in splits
                for p in (target/'runs/quality'/m/'B0'/b).glob(split+'*.jsonl')}
    if not observed <= expected:
        raise ValueError('Unreviewed calibration journal')
    return {p: sha(target/p) if (target/p).exists() else None for p in sorted(expected)}


def prerequisite_history(history, run_id, core_job, supplement_job, screen_job):
    if len(history) < 8 or len({core_job, supplement_job, screen_job}) != 3:
        raise ValueError('Core/supplement/screen prerequisites incomplete')
    supplement = screen_prerequisite_history(history[:-2], run_id, core_job, supplement_job)
    last = history[-2:]
    if ([r.get('status') for r in last] != ['started', 'complete'] or
            any(r.get('stage') != 'screen' or r.get('run_id') != run_id or
                str(r.get('slurm_job_id')) != screen_job for r in last)):
        raise ValueError('Successful screen stage required')
    if not last[-1].get('report_sha256') or not last[-1].get('admission_sha256'):
        raise ValueError('Screen completion provenance missing')
    return supplement, last[-1]


def successful_core(accounting, job):
    terminal_steps(accounting, job)
    parsed = [s.split('|') for s in accounting.splitlines() if s.strip()]
    if any(r[1] != 'COMPLETED' or r[2] != '0:0' for r in parsed):
        raise ValueError('Every core step must have completed successfully')


def planning(root, budget, calibration):
    if sha(root/COST) != COST_SHA or sha(root/COST_PROOF) != COST_PROOF_SHA:
        raise ValueError('Verified fixed extension cost sources changed')
    cost, proof = read(root/COST), read(root/COST_PROOF)
    if proof['scenario_sha256'] != COST_SHA or cost['quote_or_hard_bound'] is not False:
        raise ValueError('Historical planning provenance invalid')
    # Only calibration source journals are relevant here. Other cost-plan replica
    # absences legitimately change while independently admitted cores run.
    source = cache_inventory(root, calibration, ('calibration_extra',))
    if any(cost['source_journal_sha256_or_absent'].get(p, 'missing') != h for p, h in source.items()):
        raise ValueError('Calibration cost source journal changed')
    stage = cost['stages']['calibration']
    if stage != proof['stages']['calibration'] or stage['cells'] != 48 or stage['prospective_responses_per_run'] != 12666:
        raise ValueError('Wrong calibration cost matrix')
    values = [budget.get('spent'), budget.get('reserved'), stage['known_proxy_usd'], stage['historical_unknown_scenario_usd']]
    if any(type(v) not in (float, int) or not math.isfinite(v) or v < -1e-8 for v in values):
        raise ValueError('Invalid budget/planning value')
    if math.fsum(values) > 200:
        raise ValueError('Full calibration planning scenario exceeds current headroom')
    return {'scenario_sha256': COST_SHA, 'verification_sha256': COST_PROOF_SHA,
            'full_stage_scenario': stage, 'quote_or_hard_bound': False, 'future_cache_discount': False}


def verify_ready(root, target, run_id, spec, own, replica, protocol, calibration, screen):
    validate(spec, own, run_id, target)
    authority(root)
    if spec.get('runner_sha256') != replica['runner_sha256']:
        raise ValueError('Replica provenance changed')
    for p, digest in spec['source_sha256'].items():
        if sha(root/p) != digest:
            raise ValueError('Operational source drift: '+p)
    journal = target/'artifacts/stage_execution.jsonl'
    core_job = str(spec['core_job_id'])
    supplement_job = str(spec['supplement_job_id'])
    screen_job = str(spec['screen_job_id'])
    if own in (core_job, supplement_job, screen_job):
        raise ValueError('A distinct calibration job is required')
    supplement, screen_end = prerequisite_history(rows(journal), run_id, core_job, supplement_job, screen_job)
    if supplement['admission_sha256'] != spec['supplement_admission_sha256']:
        raise ValueError('Supplement admission provenance changed')
    reports = {str(p.relative_to(target)): sha(p) for p in (target/'artifacts/results/billiger50').iterdir() if p.is_file()}
    if reports != supplement['report_sha256'] or reports != spec['supplement_report_sha256']:
        raise ValueError('Supplement reports changed')
    screen_reports = {str(p.relative_to(target)): sha(p) for p in (target/'artifacts/results/expanded_screen').iterdir() if p.is_file()}
    if (screen_end['admission_sha256'] != spec['screen_admission_sha256'] or
            screen_reports != screen_end['report_sha256'] or screen_reports != spec['screen_report_sha256']):
        raise ValueError('Screen report/admission provenance changed')
    screen_cells = screen_component.inventory(target, run_id, screen, protocol)
    if not all(c['complete'] for c in screen_cells):
        raise ValueError('Full raw screen prerequisite required')
    supplement_cells = supplement_component.inventory(target, run_id, protocol)
    if not all(c['complete'] for c in supplement_cells):
        raise ValueError('Full raw supplement prerequisite required')
    if sha(journal) != spec['prerequisite_stage_journal_sha256'] or sha(target/'artifacts/progress.json') != spec['progress_sha256']:
        raise ValueError('Core stage/progress changed since admission')
    progress = read(target/'artifacts/progress.json')
    required = {'smoke': 'complete', 'AP1_baseline': 'complete', 'AP2_development': 'complete',
                'AP2_confirmation': 'complete', 'AP1_timing': 'complete', 'AP3_AP4_analysis': 'complete_automated',
                'supplement_billiger50': 'complete_automated', 'expanded_E2_E3_screen': 'complete'}
    if set(progress) != set(required) or any(progress[k]['status'] != v for k, v in required.items()):
        raise ValueError('Every core phase must be complete')
    if cache_inventory(target, calibration) != spec['cache_sha256_or_absent']:
        raise ValueError('Calibration cache changed since admission')
    manifest = calibration.verify()
    controls = component.inventory(target, run_id, protocol, manifest, 'route')
    if not all(c['complete'] for c in controls):
        raise ValueError('Full raw route prerequisite required')
    audited = component.inventory(target, run_id, protocol, manifest)
    if any(c['failure_rows'] or c['operations']['events'] for c in audited):
        raise ValueError('Existing calibration failure requires separate terminal review')
    budget = read(root/'artifacts/budget.json')
    cost = planning(root, budget, calibration)
    accounting = subprocess.check_output(['sacct', '-n', '-j', core_job,
        '--format=JobID,State,ExitCode,End', '-P'], text=True, env={**os.environ, 'TZ': 'UTC'})
    successful_core(accounting, core_job)
    supplement_accounting = subprocess.check_output(['sacct', '-n', '-j', supplement_job,
        '--format=JobID,State,ExitCode,End', '-P'], text=True, env={**os.environ, 'TZ': 'UTC'})
    successful_core(supplement_accounting, supplement_job)
    screen_accounting = subprocess.check_output(['sacct', '-n', '-j', screen_job,
        '--format=JobID,State,ExitCode,End', '-P'], text=True, env={**os.environ, 'TZ': 'UTC'})
    successful_core(screen_accounting, screen_job)
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
            'core_accounting_utc': accounting, 'supplement_accounting_utc': supplement_accounting, 'screen_accounting_utc': screen_accounting,
            'audited_cells': audited, 'route_cells': controls}


def audit_test_controls(target, run_id, calibration, protocol):
    checked = []
    for m in calibration.DECISION_MODELS:
        for b in calibration.BENCHMARKS:
            cases = read(target/'data'/b/'test.json')
            directory = target/'runs/quality'/m/'B0'/b
            failures = rows(directory/'test.failures.jsonl') if (directory/'test.failures.jsonl').exists() else []
            cell = audit_rows(cases, rows(directory/'test.jsonl'), failures, protocol, m, b, run_id)
            operations = inspect_operations(directory/'test.operations.jsonl', cases, failures, protocol, m, 'B0', run_id)
            if not cell['complete'] or operations['unknown_outcome_events']:
                raise ValueError('Full raw test controls required')
            checked.append(cell)
    return checked


def finish_report(target, run_id, calibration, analysis, protocol):
    manifest = calibration.verify()
    for split in ('route', 'calibration_extra'):
        if not all(c['complete'] for c in component.inventory(target, run_id, protocol, manifest, split)):
            raise ValueError('Full raw development required before fitting')
    loader = bind(analysis.load, PROJECT=target, payload=protocol.payload)
    private_analysis = types.SimpleNamespace(**{**vars(analysis), 'load': loader})
    development = bind(calibration.development, PROJECT=target, analyze=private_analysis)
    # Frozen function writes the complete development selection before ANY test load.
    entries = bind(calibration.selection, PROJECT=target, development=development)()
    selection = target/'configs/calibration_selection.json'
    if read(selection) != entries:
        raise ValueError('Frozen selection readback failed')
    selection_sha = sha(selection)
    controls = audit_test_controls(target, run_id, calibration, protocol)
    diagnostics = bind(calibration.diagnostics, analyze=private_analysis)
    bind(calibration.report, PROJECT=target, analyze=private_analysis, diagnostics=diagnostics)(entries)
    bind(calibration.cost_report, PROJECT=target, analyze=private_analysis)(manifest)
    if sha(selection) != selection_sha:
        raise ValueError('Selection changed during reporting')
    marker = target/'artifacts/expanded_calibration_complete.json'
    calibration.write_frozen(marker, {'status': 'complete_with_explicit_unavailable_fits',
        'selection_sha256': protocol.digest(entries), 'new_decisions': manifest['new_decisions'],
        'full_expanded_study_complete': False})
    outputs = [selection, marker, *sorted((target/'artifacts/results/expanded_calibration').iterdir())]
    return {'report_sha256': {str(p.relative_to(target)): sha(p) for p in outputs if p.is_file()},
            'test_control_cells': len(controls), 'selection_before_test': True}


def execute_bound(root, target, run_id, admission_path, spec, own, replica, inference, calibration, screen, protocol, analysis):
    validate(spec, own, run_id, target)
    with (root/'artifacts/operations/repetition_writer.lock').open('a') as shared, \
            (target/'artifacts/parallel_writer.lock').open('a') as exclusive:
        fcntl.flock(shared, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = target/ATTEMPT
        if attempt.exists() or any((target/p).exists() for p in (
                'configs/calibration_selection.json', 'artifacts/expanded_calibration_complete.json',
                'artifacts/results/expanded_calibration', 'artifacts/calibration_cost_forecast.json')):
            raise ValueError('Previous calibration attempt/report requires separate review')
        screen.verify()
        calibration.verify()
        checked = verify_ready(root, target, run_id, spec, own, replica, protocol, calibration, screen)
        admission_sha = sha(admission_path)
        prior_sha = sha(target/'artifacts/stage_execution.jsonl')
        attempt.mkdir(exist_ok=False)
        def log(status, **extra):
            record = {'utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
                      'stage': 'calibration', 'status': status, 'slurm_job_id': own,
                      'core_job_id': spec['core_job_id'], 'supplement_job_id': spec['supplement_job_id'], 'screen_job_id': spec['screen_job_id'],
                      'prior_stage_journal_sha256': prior_sha,
                      'admission_sha256': admission_sha, 'launcher_sha256': sha(root/SELF),
                      'parallel_policy_sha256': spec['parallel_policy_sha256'],
                      'budget_authority_sha256': APPROVAL_SHA, 'historical_unjournaled_outcomes_resolved': False, **extra}
            durable_append(attempt/'execution.jsonl', record, protocol.canonical)
            durable_append(target/'artifacts/stage_execution.jsonl', record, protocol.canonical)
        log('started', verification=checked)
        try:
            result = component.run_calibration_extra(root, target, run_id, inference, calibration, protocol, attempt/'inference')
            if not result['inference_complete']:
                raise RuntimeError('Calibration failed after all model tasks drained')
            verify(root, run_id)
            if sha(admission_path) != admission_sha:
                raise ValueError('Admission changed during execution')
            report = finish_report(target, run_id, calibration, analysis, protocol)
            log('complete', component=result, **report)
        except BaseException as exc:
            log('failed', exception_type=type(exc).__name__)
            raise
    return 0


def execute(root, run_id, admission_path):
    root = root.resolve(); target = replica_path(root, run_id)
    spec = read(admission_path); own = os.environ.get('SLURM_JOB_ID')
    validate(spec, own, run_id, target)
    if any(m in sys.modules for m in ('run', 'protocol', 'expanded_calibration', 'expanded_screen', 'expanded_protocol', 'billiger50', 'analyze', 'orchestrate')):
        raise ValueError('Fresh Python process required')
    replica = verify(root, run_id)
    if target.resolve() != target or Path(replica['replica']) != target:
        raise ValueError('Wrong replica path')
    sys.path.insert(0, str(target/'src'))
    import run, expanded_calibration, expanded_screen, billiger50, protocol, analyze, orchestrate
    if any(m.PROJECT != target for m in (run, expanded_calibration, expanded_screen, billiger50, analyze, orchestrate)):
        raise ValueError('Mixed replica modules')
    orchestrate.verify()
    billiger50.verify()
    return execute_bound(root, target, run_id, admission_path, spec, own, replica, run, expanded_calibration, expanded_screen, protocol, analyze)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', type=int, choices=(2, 3), required=True)
    parser.add_argument('--admission', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.root, args.run_id, args.admission.resolve()))
