"""One new, terminal-proof-bound missing-only continuation of job 367446.

Original runtime and all frozen requests/parsers stay byte-identical. Its execute
function is privately bound to an audited-prior loader. This permits only the
explicitly authorized, fully accounted predecessor failures, never raw gaps.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import same_condition_launch_20261016 as old
import same_condition_repetition_runtime_20261016 as runtime
import same_condition_missing_20261017 as predecessor

SELF = 'artifacts/operations/same_condition_remainder_20261017.py'
PREP = 'artifacts/verification/same_condition_remainder_readiness_20261017.json'
TERMINAL = 'artifacts/verification/same_condition_terminal_367446_20261017.json'
AUTH = 'artifacts/operations/approval_same_condition_targeted_completion_aaron_20261016.json'
AUTH_SHA = '217c7386b152bf8ac20b9a3342fd28c9919b526e3a714f69251a205ccfd39fe2'
ADMISSION = 'configs/same_condition_remainder_admission_20261017.json'
POLICY = 'configs/same_condition_remainder_scheduler_20261017.json'
ATTEMPT = 'artifacts/operations/same_condition_remainder_20261017_v1'
BATCH = 'artifacts/operations/same_condition_remainder_20261017.sbatch'
JOURNAL = 'artifacts/operations/same_condition_remainder_submission_20261017.jsonl'
NAME = 'em-identical-remainder23'
queue = types.FunctionType(old.queue.__code__, {**old.queue.__globals__, 'NAME': NAME}, argdefs=old.queue.__defaults__)


def terminal(root):
    old.require(old.sha(root / AUTH) == AUTH_SHA, 'Targeted authority drift')
    proof = old.read(root / TERMINAL)
    old.require(proof['job'] == '367446' and proof['all_steps_terminal'] is True
                and proof['independent_scoped_raw_and_credit_verified'] is True
                and proof['authority_sha256'] == AUTH_SHA
                and proof['readiness_sha256'] == old.sha(root / predecessor.PREP), 'Missing terminal proof')
    for p in proof['run_proofs'].values():
        old.require(old.sha(root / p['path']) == p['sha256'], 'Terminal run proof drift')
    return proof


def capture_prior(root, target, rid, inference, protocol, expanded, schema, journal_dir):
    proof = terminal(root)
    old.require(journal_dir == target / ATTEMPT / 'inference' and not journal_dir.exists(), 'Exclusive bounded attempt required')
    result = old.read(root / proof['run_proofs'][str(rid)]['path'])
    old.require(result['run_id'] == rid and result['remaining'] > 0, 'No missing work in this run')
    for rel, h in result['attempt_sha256'].items():
        old.require(old.sha(root / rel) == h, 'Predecessor history drift')
    cases, protected = runtime.sources(root, target, rid, inference, protocol, expanded, schema)
    journals = {str(p.relative_to(target)): runtime.prefix(p) for p in runtime.paths(target, runtime.matrix(rid))}
    checked = runtime.inventory(target, rid, cases, protocol, expanded, schema)
    old.require(checked == result['cells'] and journals == result['journals']
                and protected == result['protected_sha256'], 'Terminal-scoped raw/source drift')
    old.require(journals == {p: runtime.prefix(target / p) for p in journals}, 'Concurrent writer')
    runtime.unchanged(protected)
    return dict(run_id=rid, stage=runtime.STAGE, replica=str(target), journal_dir=str(journal_dir.relative_to(target)),
                cells=checked, protected_sha256=protected, journals=journals,
                prior_responses=sum(c['responses'] for c in checked), admission_granted=False)


# Only this private function resolves capture_prior to the proof-bound loader.
# All real raw/input/request/parser/run/source checks and reconciliation survive.
def execute_runtime(*args):
    bound = types.FunctionType(runtime.execute.__code__,
        {**runtime.execute.__globals__, 'capture_prior': capture_prior}, argdefs=runtime.execute.__defaults__)
    return bound(*args)


def protect(root, prep):
    predecessor.protect(root, old.read(root / predecessor.PREP))
    for rel, h in prep['sources'].items():
        old.require(old.file_state(root / rel) == h, 'Continuation source drift: ' + rel)
    for run in prep['terminal']['run_proofs'].values():
        result = old.read(root / run['path'])
        for rel, h in result['attempt_sha256'].items():
            old.require(old.sha(root / rel) == h, 'Predecessor attempt drift')
        target = root / f"replications/run{result['run_id']:02d}"
        for rel, info in result['journals'].items():
            p = target / rel; raw = p.read_bytes() if p.exists() else b''
            old.require(len(raw) >= info['bytes'] and hashlib.sha256(raw[:info['bytes']]).hexdigest() == info['sha256']
                        and (not info['exists'] or p.exists()), 'Predecessor successful/failure prefix drift')


def foreign_credits(root, prep):
    budget = old.budget_check(root)
    before = prep['terminal']['budget']
    for key, value in before['inference_credits'].items():
        old.require(budget['inference_credits'][key] == value, 'Foreign or predecessor credit drift')
    predecessor.foreign_credits(root, old.read(root / predecessor.PREP))


def gate(root, spec, running=False, rid=None):
    old.require(spec == old.read(root / ADMISSION) and spec['admitted'] is True
                and spec['admission_path'] == ADMISSION and spec['scope'] == runtime.STAGE
                and spec['run_ids'] == [2, 3] and spec['max_invocations_per_run'] == 1
                and spec['project_cap_usd'] == 250 and spec['new_selection'] is False
                and spec['calibration_calls'] == 0 and spec['approval_sha256'] == AUTH_SHA
                and spec['readiness_sha256'] == old.sha(root / PREP)
                and spec['policy_sha256'] == old.sha(root / POLICY), 'Real exact admission required')
    prep = old.read(root / PREP)
    old.require(terminal(root) == prep['terminal'] and spec['total_missing'] == sum(prep['terminal']['missing'].values()), 'Scope drift')
    protect(root, prep); foreign_credits(root, prep)
    old.budget_check(root, 0 if running else 16)
    history = old.steps(root, prep['terminal']['history_steps'], prep['terminal']['history_steps'])
    sched = queue(root, old.read(root / POLICY), running)
    for run in ((rid,) if rid else (2, 3)):
        old.require(not (root / f'replications/run{run:02d}' / ATTEMPT).exists(), 'Attempt already exists')
    return dict(utc=old.now(), history_steps=history, scheduler=sched, budget=old.budget_check(root))


def inventory(root, rid):
    target = root / f'replications/run{rid:02d}'
    return capture_prior(root, target, rid, *old.load_modules(target), target / ATTEMPT / 'inference')


def child_inventory(root, rid):
    p = subprocess.run([sys.executable, str(root / SELF), '--root', str(root), 'inventory', '--run-id', str(rid)], capture_output=True, text=True)
    old.require(p.returncode == 0, 'Child inventory failed: ' + p.stderr)
    return json.loads(p.stdout)


def prepare(root):
    with old.locks(root):
        proof = terminal(root); old.steps(root, proof['history_steps'], proof['history_steps'])
        queue(root); old.budget_check(root, 16)
        old.require(old.read(root / 'artifacts/budget.json') == proof['budget'], 'Budget changed since terminal audit')
        priors = {str(rid): child_inventory(root, rid) for rid in (2, 3)}
        sources = set(old.read(root / predecessor.PREP)['sources'])
        sources.update((SELF, BATCH, TERMINAL, AUTH, old.PREP, old.ADMISSION, old.POLICY,
                        'artifacts/verification/audit_same_condition_missing_terminal_20261017.py',
                        predecessor.SELF, predecessor.PREP, predecessor.ADMISSION, predecessor.POLICY,
                        'artifacts/verification/same_condition_missing_jev_failure_20261017T0803Z.json',
                        'artifacts/verification/same_condition_missing_jev_run3_failure_20261017T0903Z.json'))
        sources.update(p['path'] for p in proof['run_proofs'].values())
        prep = dict(utc=old.now(), terminal=proof, priors=priors,
                    sources={rel: old.sha(root / rel) for rel in sorted(sources)},
                    ordering='Run2 then Run3 to avoid simultaneous requests to the same provider',
                    max_invocations_per_run=1, admitted=False)
        protect(root, prep); foreign_credits(root, prep)
        old.save(root / PREP, prep)
        print(json.dumps({'prepared_only': True, 'missing': proof['missing']}))


def preflight(root, output):
    with old.locks(root):
        spec = old.read(root / ADMISSION); state = gate(root, spec)
        prep = old.read(root / PREP)
        for rid in (2, 3):
            old.require(child_inventory(root, rid) == prep['priors'][str(rid)], 'Exact prior changed')
        protect(root, prep)
        old.save(output, dict(utc=old.now(), admission_sha256=old.sha(root / ADMISSION),
                             checks=state, full_scoped_raw_readback=True))


def execute(root, rid):
    spec = old.read(root / ADMISSION)
    old.require(os.environ.get('SLURM_JOB_ID') == spec['slurm_job_id'], 'Wrong real admitted job')
    target = root / f'replications/run{rid:02d}'; modules = old.load_modules(target)
    with old.locks(root, (rid,)):
        state = gate(root, spec, True, rid)
        prior = capture_prior(root, target, rid, *modules, target / ATTEMPT / 'inference')
        old.require(prior == old.read(root / PREP)['priors'][str(rid)], 'Prior changed after release')
        attempt = target / ATTEMPT; attempt.mkdir(); runtime.sync_dir(attempt.parent)
        old.save(attempt / 'prior.json', prior); old.save(attempt / 'start_gate.json', state)
        credit = old.credit_class(root)(root, f'same-remainder-{rid}-' + spec['slurm_job_id'],
            dict(prior_sha256=old.sha(attempt / 'prior.json'), run_id=rid, stage=runtime.STAGE,
                 journal_dir=str((attempt / 'inference').relative_to(target)), admission_sha256=old.sha(root / ADMISSION)), chunk=8)
        old.save(attempt / 'started.json', dict(utc=old.now(), run_id=rid, job=spec['slurm_job_id'], credit=credit.summary(),
                 prior_sha256=old.sha(attempt / 'prior.json'), admission_sha256=old.sha(root / ADMISSION)))
        try:
            result = execute_runtime(root, target, rid, *modules, attempt / 'prior.json', old.sha(attempt / 'prior.json'), attempt / 'inference', credit)
            old.save(attempt / 'execution_result.json', result)
            protect(root, old.read(root / PREP))
            closed = runtime.reconcile(root, target, rid, *modules, attempt / 'prior.json', old.sha(attempt / 'prior.json'), attempt / 'inference', credit)
            old.save(attempt / 'credit_reconciliation.json', closed)
            foreign_credits(root, old.read(root / PREP))
            old.require(result['inference_complete'] and closed['inference_complete'], 'Still missing; new terminal review required')
            old.save(attempt / 'inference_complete.json', dict(utc=old.now(), run_id=rid, independent_report_and_slurm_steps_pending=True,
                result_sha256=old.sha(attempt / 'execution_result.json'), credit_sha256=old.sha(attempt / 'credit_reconciliation.json')))
        except Exception as exc:
            old.save(attempt / 'failed.json', dict(utc=old.now(), exception_type=type(exc).__name__, credit=credit.summary(), hold_retained=not credit.closed))
            raise


def log(root, event, **data):
    with (root / JOURNAL).open('a') as f:
        f.write(json.dumps(dict(utc=old.now(), event=event, **data)) + '\n'); f.flush(); os.fsync(f.fileno())


def full_gate(root, kind):
    p = root / f'artifacts/verification/same_condition_remainder_{kind}_20261017.json'
    preflight(root, p)
    log(root, kind + '_passed', sha256=old.sha(p), admission_sha256=old.sha(root / ADMISSION))


def submit(root):
    with old.locks(root):
        prep = old.read(root / PREP); protect(root, prep); foreign_credits(root, prep)
        jobs = queue(root); old.steps(root, prep['terminal']['history_steps'], prep['terminal']['history_steps']); old.budget_check(root, 16)
        old.require(not (root / JOURNAL).exists() and not (root / ADMISSION).exists(), 'Already submitted')
        for rid in (2, 3):
            old.require(not (root / f'replications/run{rid:02d}' / ATTEMPT).exists(), 'Attempt exists')
        with (root / JOURNAL).open('x') as f:
            f.flush(); os.fsync(f.fileno())
        runtime.sync_dir((root / JOURNAL).parent)
        log(root, 'held_submission_intent', batch_sha256=old.sha(root / BATCH))
        job = old.cmd('sbatch', '--hold', '--parsable', '--partition=cpu', '--cpus-per-task=4', '--mem=24G', '--time=2-00:00:00',
            '--job-name=' + NAME, '--output=' + str(root / 'logs/same-remainder-%j.log'), '--chdir=' + str(root), str(root / BATCH)).strip()
        old.require(job.isdigit(), 'Unknown submission outcome; never blindly resubmit')
        log(root, 'held_submission_returned', job=job)
        jobs[job] = dict(name=NAME, batch_script_sha256=old.sha(root / BATCH), dependency='(null)')
        old.save(root / POLICY, dict(jobs=jobs, scope=runtime.STAGE, run_ids=[2, 3], overlap_cells=0))
        spec = dict(admitted=True, scope=runtime.STAGE, run_ids=[2, 3], max_invocations_per_run=1, project_cap_usd=250,
                    new_selection=False, calibration_calls=0, slurm_job_id=job, admission_path=ADMISSION,
                    readiness_sha256=old.sha(root / PREP), policy_sha256=old.sha(root / POLICY),
                    approval_sha256=AUTH_SHA, total_missing=sum(prep['terminal']['missing'].values()))
        old.save(root / ADMISSION, spec)
        old.save(root / 'artifacts/operations/same_condition_remainder_job_20261017.json', dict(job=job, admission=ADMISSION))
    full_gate(root, 'held_gate')
    print(json.dumps(dict(job=job, held=True)))


def release(root):
    events = runtime.rows(root / JOURNAL)
    old.require(events[-1]['event'] == 'held_gate_passed' and events[-1]['admission_sha256'] == old.sha(root / ADMISSION), 'Held gate absent')
    full_gate(root, 'prerelease_gate')
    spec = old.read(root / ADMISSION); log(root, 'release_intent', job=spec['slurm_job_id'])
    subprocess.run(['scontrol', 'release', spec['slurm_job_id']], check=True)
    log(root, 'release_returned', metadata=old.cmd('scontrol', 'show', 'job', spec['slurm_job_id'], '-o'))
    print(json.dumps(dict(job=spec['slurm_job_id'], released=True)))


def launch(root):
    old.require(os.environ.get('SLURM_JOB_ID') == old.read(root / ADMISSION)['slurm_job_id'], 'Wrong job')
    codes = [subprocess.run([sys.executable, '-u', str(root / SELF), '--root', str(root), 'execute', '--run-id', str(rid)]).returncode for rid in (2, 3)]
    old.require(codes == [0, 0], 'Missing continuation incomplete; no automatic retry')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('action', choices=('prepare', 'inventory', 'submit', 'release', 'execute', 'launch'))
    p.add_argument('--run-id', type=int, choices=(2, 3)); a = p.parse_args(); root = a.root.resolve()
    if a.action == 'inventory':
        print(json.dumps(inventory(root, a.run_id)))
    elif a.action == 'execute':
        execute(root, a.run_id)
    else:
        globals()[a.action](root)
