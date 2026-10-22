"""One admitted fresh confirmation: development, durable choice, priced tests.

The initial admission funds only the full development planning scenario. Actual
selected missing tests need another current gate under the same writer locks.
A stopped attempt, including a frozen choice without tests, needs a separately
reviewed continuation. This launcher never submits, releases or retries jobs.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from authorized_recovery_200 import APPROVAL, APPROVAL_SHA, PINNED, SELF as BINDING, authority, make_wrapper
from drained_inference import durable_append
import fresh_confirmation_development_drained as development
import fresh_confirmation_selection as freeze
import fresh_confirmation_test_drained as tests
import fresh_confirmation_selected_cost as selected_cost
import fresh_screen_drained as screen_component
import fresh_supplement_drained as supplement_component
from run_fresh_stability_drained import prerequisite_history, successful_core
from repeated_runs import replica_path, sha, verify

SELF = 'artifacts/operations/run_fresh_confirmation_drained.py'
ATTEMPT = 'artifacts/fresh_confirmation_drained_v1'
COST = 'artifacts/verification/fresh_confirmation_cost_scenarios_20261015.json'
COST_SHA = '0ef743cfafd47650819227ba0e4bebecaf18cb25da9bf101f9e5f986621cbfae'
COST_PROOF = 'artifacts/verification/fresh_confirmation_cost_verification_20261015.json'
COST_PROOF_SHA = '660654bc04adbb18d441abbda30dbf6a08ba0a86b650324e95a3b774eb32707b'
CORE_S1 = {'jev', 'clef_flash', 'luna_decisions', 'luna'}
CORE_OUTPUTS = {'configs/selected_variants.json'} | {'artifacts/results/'+name for name in (
    'main_metrics.csv', 'ablations.csv', 'timing.csv', 'reliability.json', 'cascades.json',
    'paired_intervals.json', 'main_metrics.json', 'summary.json', 'audit_blind_160.csv',
    'audit_second_reviewer_40.csv', 'audit_key_private.json')}
SUFFIXES = ('.jsonl', '.failures.jsonl', '.operations.jsonl', '.executions.jsonl')
REQUIRED = {SELF, BINDING, APPROVAL} | {f'artifacts/operations/{n}.py' for n in (
    *PINNED, 'fresh_confirmation_development_drained', 'fresh_confirmation_selection',
    'fresh_confirmation_test_drained', 'fresh_confirmation_selected_cost',
    'fresh_confirmation_cost_plan', 'fresh_screen_drained', 'fresh_supplement_drained',
    'fresh_stability_drained', 'run_fresh_stability_drained', 'run_fresh_screen_drained',
    'run_fresh_supplement_drained', 'diagnose_fresh_baseline', 'operational_journal',
    'repeated_runs', 'resume_schema_once')}


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def validate(spec, own, run_id, target):
    expected = {'admitted': True, 'stage': 'confirmation', 'run_id': run_id,
        'slurm_job_id': own, 'replica': str(target), 'budget_cap_usd': 200,
        'budget_authority_sha256': APPROVAL_SHA, 'max_invocations_per_model_per_phase': 1,
        'historical_unknown_outcomes_disclosed': True, 'two_phase_budget_gate': True}
    if (not own or type(run_id) is not int or run_id not in (2, 3)
            or any(spec.get(k) != v for k, v in expected.items())):
        raise ValueError('Actual reviewed two-phase confirmation admission required')
    if not REQUIRED <= spec.get('source_sha256', {}).keys():
        raise ValueError('Executable provenance incomplete')


def cache_inventory(target, confirmation, protocol):
    cells = [(m, v, b, 'prompt') for m, v, b in development.matrix(confirmation)]
    cells += [(m, v, b, 'test') for m in protocol.MODELS
              for v in [*confirmation.PROMPT_VARIANTS, *confirmation.SERIALIZATIONS]
              for b in [*protocol.BENCHMARKS, 'billiger-de-50un']]
    expected = {f'runs/quality/{m}/{v}/{b}/{split}{s}' for m, v, b, split in cells for s in SUFFIXES}
    observed = {str(p.relative_to(target)) for m, v, b, split in cells
                for p in (target/'runs/quality'/m/v/b).glob(split+'*.jsonl')}
    if not observed <= expected:
        raise ValueError('Unreviewed confirmation journal')
    expected |= {f'data/{b}/{split}.json' for _, _, b, split in cells}
    return {p: sha(target/p) if (target/p).exists() else None for p in sorted(expected)}


def planning(root, budget, confirmation, protocol):
    if sha(root/COST) != COST_SHA or sha(root/COST_PROOF) != COST_PROOF_SHA:
        raise ValueError('Verified development cost provenance changed')
    cost, proof = read(root/COST), read(root/COST_PROOF)
    if proof['scenario_sha256'] != COST_SHA or cost['quote_or_hard_bound'] is not False:
        raise ValueError('Historical planning provenance invalid')
    protected = cost['protected_sha256_or_absent']
    # Run1 development sources only: evolving selected tests and replica caches
    # are not cost sources and their old absences are not an admission.
    paths = {f'runs/quality/{m}/{v}/{b}/prompt{s}'
             for m, v, b in development.matrix(confirmation) for s in SUFFIXES}
    paths |= {f'data/{b}/prompt.json' for b in protocol.BENCHMARKS}
    if any(protected.get(p, 'missing') != (sha(root/p) if (root/p).exists() else None) for p in paths):
        raise ValueError('Run1 development cost source changed')
    stage = cost['development_alternatives_not_additive']['conditional_successful_core']
    if stage['cells'] != 1088 or stage['prospective_responses'] != 217600:
        raise ValueError('Wrong full development scenario')
    checked = selected_cost.budget_review(stage, budget)
    if not checked['scenario_fits_current_headroom']:
        raise ValueError('Full development scenario exceeds current headroom')
    return {'scenario_sha256': COST_SHA, 'verification_sha256': COST_PROOF_SHA,
            'development': stage, **checked, 'test_admission_granted': False,
            'unknown_choice_envelope_used_as_admission': False, 'quote_or_hard_bound': False}


def accounting(job):
    result = subprocess.check_output(['sacct', '-n', '-j', job,
        '--format=JobID,State,ExitCode,End', '-P'], text=True, env={**os.environ, 'TZ': 'UTC'})
    successful_core(result, job)
    return result


def prior_stages(target, history, run_id, spec):
    prior = spec.get('previous_completed_stages', [])
    names = [r.get('stage') for r in prior]
    if names not in ([], ['calibration'], ['calibration', 'stability']):
        raise ValueError('Unreviewed intervening stages; calibration must precede stability')
    checked = []
    for info in reversed(prior):
        last = history[-2:]
        stage, job = info['stage'], str(info['slurm_job_id'])
        if (len(last) != 2 or [r.get('status') for r in last] != ['started', 'complete']
                or any(r.get('stage') != stage or r.get('run_id') != run_id
                       or str(r.get('slurm_job_id')) != job for r in last)
                or any(not info.get(k) or last[-1].get(k) != info[k]
                       for k in ('admission_sha256', 'report_sha256'))):
            raise ValueError('Intervening stage provenance incomplete')
        marker = f'artifacts/expanded_{stage}_complete.json'
        status = 'complete_with_explicit_unavailable_fits' if stage == 'calibration' else 'complete'
        if marker not in info['report_sha256'] or read(target/marker).get('status') != status:
            raise ValueError('Intervening completion marker invalid')
        for p, digest in info['report_sha256'].items():
            path = (target/p).resolve()
            if not path.is_relative_to(target) or sha(path) != digest:
                raise ValueError('Intervening report changed')
        checked.append({'stage': stage, 'slurm_job_id': job, 'accounting_utc': accounting(job)})
        history = history[:-2]
    return history, list(reversed(checked))


def current_scheduler(root, target, run_id, spec, own):
    policy = make_wrapper(root, 'core').review(root, root/spec['parallel_policy'], own)
    if (policy['policy_sha256'] != spec['parallel_policy_sha256']
            or not any(j.get('self') and j['id'] == own for j in policy['reviewed_jobs'])):
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
                raise ValueError('Another writer targets this replica')
    if dependencies != spec['reviewed_dependencies']:
        raise ValueError('All queue joins must match current admission')
    return {'policy': policy, 'dependencies': dependencies}


def verify_ready(root, target, run_id, spec, own, replica, protocol, confirmation, screen, started_sha=None):
    validate(spec, own, run_id, target)
    authority(root)
    if spec.get('runner_sha256') != replica['runner_sha256']:
        raise ValueError('Replica provenance changed')
    for p, digest in spec['source_sha256'].items():
        path = (root/p).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError('Operational source drift: '+p)
    journal = target/'artifacts/stage_execution.jsonl'
    if sha(journal) != (started_sha or spec['prerequisite_stage_journal_sha256']):
        raise ValueError('Stage journal changed since reviewed boundary')
    history = rows(journal)
    if started_sha:
        own_start = history.pop()
        if any(own_start.get(k) != v for k, v in
               {'stage': 'confirmation', 'status': 'started', 'run_id': run_id, 'slurm_job_id': own}.items()):
            raise ValueError('Wrong active confirmation stage')
    history, previous = prior_stages(target, history, run_id, spec)
    core, supplement_job, screen_job = (str(spec[k]) for k in ('core_job_id', 'supplement_job_id', 'screen_job_id'))
    jobs = [own, core, supplement_job, screen_job, *(r['slurm_job_id'] for r in previous)]
    if len(set(jobs)) != len(jobs):
        raise ValueError('Every prerequisite and current stage needs a distinct job')
    supplement, screen_end = prerequisite_history(history, run_id, core, supplement_job, screen_job)
    core_end = [r for r in history if r.get('stage') == 'core'][-1]
    core_admission = (root/spec['core_admission']).resolve()
    if (not core_admission.is_relative_to(root)
            or sha(core_admission) != spec['core_admission_sha256']
            or core_end.get('admission_sha256') != spec['core_admission_sha256']
            or protocol.digest(core_end) != spec['core_stage_end_sha256']):
        raise ValueError('Core completion/admission provenance changed')
    admitted_core = read(core_admission)
    if any(admitted_core.get(k) != v for k, v in
           {'admitted': True, 'run_id': run_id, 'stage': 'core', 'slurm_job_id': core}.items()):
        raise ValueError('Core admission identity mismatch')
    if (set(spec['core_report_sha256']) != CORE_OUTPUTS
            or any(sha(target/p) != h for p, h in spec['core_report_sha256'].items())):
        raise ValueError('Core reports changed or incomplete')
    for end, folder, key in ((supplement, 'billiger50', 'supplement'), (screen_end, 'expanded_screen', 'screen')):
        reports = {str(p.relative_to(target)): sha(p) for p in (target/'artifacts/results'/folder).iterdir() if p.is_file()}
        if (end['admission_sha256'] != spec[key+'_admission_sha256']
                or reports != end['report_sha256'] or reports != spec[key+'_report_sha256']):
            raise ValueError('Prerequisite report/admission provenance changed')
    if sha(target/'artifacts/progress.json') != spec['progress_sha256']:
        raise ValueError('Core progress changed')
    required = {'smoke': 'complete', 'AP1_baseline': 'complete', 'AP2_development': 'complete',
        'AP2_confirmation': 'complete', 'AP1_timing': 'complete', 'AP3_AP4_analysis': 'complete_automated',
        'supplement_billiger50': 'complete_automated', 'expanded_E2_E3_screen': 'complete'}
    progress = read(target/'artifacts/progress.json')
    if set(progress) != set(required) or any(progress[k]['status'] != v for k, v in required.items()):
        raise ValueError('Full core/supplement/screen progress required')
    screen.verify(); confirmation.verify()
    if not all(c['complete'] for c in screen_component.inventory(target, run_id, screen, protocol)):
        raise ValueError('Full raw screen prerequisite required')
    if not all(c['complete'] for c in supplement_component.inventory(target, run_id, protocol)):
        raise ValueError('Full raw supplement prerequisite required')
    dev = development.inventory(target, run_id, confirmation, protocol)
    reused = [c for c in dev if c['variant'] == 'B0' or (c['variant'] == 'S1' and c['model'] in CORE_S1)]
    if len(reused) != 88 or not all(c['complete'] for c in reused):
        raise ValueError('Full own core B0 and S1 development required for cost credit')
    controls = {(m, 'B0', b) for m in protocol.MODELS for b in [*protocol.BENCHMARKS, 'billiger-de-50un']}
    if not all(c['complete'] for c in tests.inventory(target, run_id, confirmation, protocol, controls)):
        raise ValueError('Full raw own B0 test controls required')
    if started_sha is None:
        caches = cache_inventory(target, confirmation, protocol)
        if caches != spec['cache_sha256_or_absent']:
            raise ValueError('Confirmation cache changed since admission')
        if any((c['responses'] or c['failure_rows'] or c['operations']['events'])
               and c['variant'] not in ('B0', 'S1') for c in dev):
            raise ValueError('Previous expanded development needs terminal review')
        if any(h is not None and '/test' in p and p.startswith('runs/')
               and p.split('/')[3] not in ('B0', 'S1') for p, h in caches.items()):
            raise ValueError('Previous expanded test journal needs terminal review')
        cost = planning(root, read(root/'artifacts/budget.json'), confirmation, protocol)
    else:
        if not all(c['complete'] for c in dev):
            raise ValueError('Full raw development required before test gate')
        cost = None  # Actual own selected costs are checked separately, immediately before test.
    accounts = {job: accounting(job) for job in (core, supplement_job, screen_job)}
    return {'cost': cost, 'prerequisite_accounting_utc': accounts,
            'previous_completed_stages': previous, **current_scheduler(root, target, run_id, spec, own)}


def execute_bound(root, target, run_id, admission_path, spec, own, replica,
                  inference, confirmation, expanded, screen, protocol, analysis):
    validate(spec, own, run_id, target)
    with (root/'artifacts/operations/repetition_writer.lock').open('a') as shared, \
            (target/'artifacts/parallel_writer.lock').open('a') as exclusive:
        fcntl.flock(shared, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        forbidden = (ATTEMPT, freeze.ATTEMPT, tests.ATTEMPT, tests.REPORT_ATTEMPT, tests.MARKER, tests.REPORT_DIR,
                     *freeze.OUTPUTS, 'artifacts/expanded_confirmation_cost_forecast.json')
        if any((target/p).exists() for p in forbidden):
            raise ValueError('Prior confirmation attempt/output requires separate review')
        admission_sha = sha(admission_path)
        if read(admission_path) != spec:
            raise ValueError('Admission differs from loaded specification')
        checked = verify_ready(root, target, run_id, spec, own, replica, protocol, confirmation, screen)
        prior_sha = sha(target/'artifacts/stage_execution.jsonl')
        attempt = target/ATTEMPT; attempt.mkdir(exist_ok=False)
        def log(status, **extra):
            record = {'utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
                'stage': 'confirmation', 'status': status, 'slurm_job_id': own,
                'prior_stage_journal_sha256': prior_sha, 'admission_sha256': admission_sha,
                'launcher_sha256': sha(root/SELF), 'parallel_policy_sha256': spec['parallel_policy_sha256'],
                'budget_authority_sha256': APPROVAL_SHA, 'historical_unjournaled_outcomes_resolved': False, **extra}
            durable_append(attempt/'execution.jsonl', record, protocol.canonical)
            durable_append(target/'artifacts/stage_execution.jsonl', record, protocol.canonical)
        log('started', verification=checked)
        started_sha = sha(target/'artifacts/stage_execution.jsonl')
        args = (root, target, run_id, confirmation, expanded, analysis, protocol)
        phase = 'development'
        try:
            result = development.run_development(root, target, run_id, inference, confirmation,
                expanded, analysis, protocol, attempt/'development')
            if not result['development_complete']:
                raise RuntimeError('Development failed after all model tasks drained')
            phase = 'freeze'
            freeze.freeze_development(*args, attempt/'development/models.jsonl')
            phase = 'selected_test_gate'
            verify(root, run_id)
            if sha(admission_path) != admission_sha:
                raise ValueError('Admission changed during development')
            current = verify_ready(root, target, run_id, spec, own, replica, protocol, confirmation, screen, started_sha)
            priced = selected_cost.review(*args)
            # Sources cannot move under the replica lock. Recheck live policy/joins
            # after the potentially long raw replay, then reread the shared ledger.
            current.update(current_scheduler(root, target, run_id, spec, own))
            headroom = selected_cost.require_headroom(root, priced)
            if sha(admission_path) != admission_sha or sha(target/'artifacts/stage_execution.jsonl') != started_sha:
                raise ValueError('Admission or active stage changed before test')
            receipt = {'run_id': run_id, 'slurm_job_id': own, 'admission_sha256': admission_sha,
                       'current_gate': current, 'selected_test_costs': priced, 'fresh_budget_gate': headroom}
            freeze.durable_write(attempt/'selected_test_gate.json', (json.dumps(receipt, indent=2)+'\n').encode())
            phase = 'test'
            result = tests.run_selected_tests(root, target, run_id, inference, confirmation, expanded, analysis, protocol)
            if not result['test_complete']:
                raise RuntimeError('Selected test failed after all model tasks drained')
            phase = 'report'
            verify(root, run_id)
            if sha(admission_path) != admission_sha:
                raise ValueError('Admission changed during tests')
            report = tests.finish_report(*args)
            log('complete', report_sha256=report,
                selected_test_gate_sha256=sha(attempt/'selected_test_gate.json'),
                selection_receipt_sha256=sha(target/freeze.RECEIPT))
        except BaseException as exc:
            log('failed', phase=phase, exception_type=type(exc).__name__,
                selection_preserved=(target/freeze.RECEIPT).exists())
            raise
    return 0


def execute(root, run_id, admission_path):
    root = root.resolve(); target = replica_path(root, run_id)
    spec = read(admission_path); own = os.environ.get('SLURM_JOB_ID')
    validate(spec, own, run_id, target)
    names = ('run', 'protocol', 'expanded_confirmation', 'expanded_screen', 'expanded_protocol',
             'billiger50', 'analyze', 'orchestrate')
    if any(m in sys.modules for m in names):
        raise ValueError('Fresh Python process required')
    replica = verify(root, run_id)
    if target.resolve() != target or Path(replica['replica']) != target:
        raise ValueError('Wrong replica path')
    sys.path.insert(0, str(target/'src'))
    import run, expanded_confirmation, expanded_protocol, expanded_screen, billiger50, protocol, analyze, orchestrate
    if any(m.PROJECT != target for m in (run, expanded_confirmation, expanded_screen, billiger50, analyze, orchestrate)):
        raise ValueError('Mixed replica modules')
    orchestrate.verify(); billiger50.verify()
    return execute_bound(root, target, run_id, admission_path, spec, own, replica, run,
        expanded_confirmation, expanded_protocol, expanded_screen, protocol, analyze)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', type=int, choices=(2, 3), required=True)
    parser.add_argument('--admission', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.root, args.run_id, args.admission.resolve()))
