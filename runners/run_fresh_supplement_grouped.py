"""Bounded grouped supplement wrapper; never submits, releases or retries jobs.

Only canonical successful smoke/core histories are supported. The actual
multi-recovery histories remain rejected until separately adapted after verified
Core success. Existing per-call wrappers and grouped helpers are unchanged.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys

from authorized_recovery_200 import APPROVAL_SHA, authority
from drained_inference import durable_append
import fresh_supplement_grouped as grouped
import fresh_schema_selection as files
from leased_budget import Credit
from repeated_runs import replica_path, sha, verify
from rules_drained import bind
import run_fresh_supplement_drained as old
import run_fresh_confirmation_drained as common

SELF = 'artifacts/operations/run_fresh_supplement_grouped.py'
ATTEMPT = 'artifacts/supplement_grouped_v1'
REPORT_DIR = 'artifacts/results/billiger50'
MARKER = 'artifacts/supplement_grouped_complete.json'
REPORT_NAMES = ('metrics.csv', 'metrics.json', 'paired_intervals.json',
                'reliability.json', 'historical_reference.json', 'comparison.md')
REQUIRED = old.REQUIRED | common.REQUIRED | {SELF} | {f'artifacts/operations/{n}.py' for n in (
    'run_fresh_confirmation_drained', 'run_fresh_supplement_drained',
    'fresh_supplement_grouped', 'fresh_rules_grouped', 'fresh_schema_selection',
    'grouped_inference', 'leased_budget', 'rules_drained')}
read = old.read


def validate(spec, own, rid, target):
    expected = dict(admitted=True, stage='supplement', run_id=rid, slurm_job_id=own,
        replica=str(target), budget_cap_usd=200, budget_authority_sha256=APPROVAL_SHA,
        max_invocations=1, historical_unknown_outcomes_disclosed=True,
        accounting_transport='bounded_process_credit', journal_transport='group_commit_v1',
        credit_chunk_usd=8, checkpoint_reservations=512,
        history_adapter='canonical_success_only_v1')
    if (not own or type(rid) is not int or rid not in (2, 3)
            or any(spec.get(k) != v for k, v in expected.items())):
        raise ValueError('Actual reviewed grouped supplement admission required')
    if not REQUIRED <= spec.get('source_sha256', {}).keys():
        raise ValueError('Executable provenance incomplete')


def sources(root, spec):
    authority(root)
    for p, h in spec['source_sha256'].items():
        if sha(files.own_path(root, p)) != h:
            raise ValueError('Operational source drift: '+p)


def prerequisites(root, target, rid, spec, protocol, started_sha=None):
    path = target/'artifacts/stage_execution.jsonl'
    grouped.prefix(path)
    if sha(path) != (started_sha or spec['core_stage_journal_sha256']):
        raise ValueError('Core stage history changed')
    history = old.rows(path)
    raw = path.read_bytes().splitlines(keepends=True)
    if started_sha:
        own = history.pop(); raw.pop()
        if any(own.get(k) != v for k, v in dict(stage='supplement', status='started',
                run_id=rid, slurm_job_id=spec['slurm_job_id']).items()):
            raise ValueError('Wrong own active stage')
    if (len(history) != 4 or [r.get('stage') for r in history] != ['smoke','smoke','core','core']
            or [r.get('status') for r in history] != ['started','complete','started','complete']
            or sha_bytes(b''.join(raw)) != spec['core_stage_journal_sha256']):
        raise ValueError('Unreviewed entire core history; recovery adapter is not admitted')
    infos = spec['prerequisites']
    if [r.get('stage') for r in infos] != ['smoke','core']:
        raise ValueError('Whole smoke/core prerequisite provenance required')
    accounts = {}; jobs = {spec['slurm_job_id']}
    for i, info in enumerate(infos):
        stage, job = info['stage'], str(info['slurm_job_id'])
        pair = history[2*i:2*i+2]
        if job in jobs or (stage == 'core' and job != str(spec['core_job_id'])):
            raise ValueError('Distinct exact prerequisite jobs required')
        jobs.add(job)
        if (any(type(r.get('run_id')) is not int or r['run_id'] != rid
                or str(r.get('slurm_job_id')) != job for r in pair)
                or protocol.digest(pair[-1]) != info['stage_end_sha256']):
            raise ValueError('Prerequisite stage identity/end changed')
        admission = files.own_path(root, info['admission'])
        if (sha(admission) != info['admission_sha256']
                or any(r.get('admission_sha256') != info['admission_sha256'] for r in pair)
                or any(read(admission).get(k) != v for k, v in dict(
                    admitted=True, stage=stage, run_id=rid, slurm_job_id=job).items())):
            raise ValueError('Prerequisite admission provenance changed')
        reports = info['report_sha256']
        if set(reports) != (common.CORE_OUTPUTS if stage == 'core' else set()):
            raise ValueError('Incomplete prerequisite report inventory')
        if stage == 'core' and pair[-1].get('report_sha256') != reports:
            raise ValueError('Core report/end provenance mismatch')
        for p, h in reports.items():
            if sha(files.own_path(target, p)) != h:
                raise ValueError('Core reports changed')
        accounts[job] = common.accounting(job)
    progress = read(target/'artifacts/progress.json')
    expected = dict(smoke='complete', AP1_baseline='complete', AP2_development='complete',
        AP2_confirmation='complete', AP1_timing='complete', AP3_AP4_analysis='complete_automated')
    if (set(progress) != set(expected) or any(progress[k]['status'] != v for k, v in expected.items())
            or sha(target/'artifacts/progress.json') != spec['progress_sha256']):
        raise ValueError('Full actual core progress required')
    return dict(accounting_utc=accounts, history_adapter='canonical_success_only_v1',
                current_recovery_histories_admitted=False)


def sha_bytes(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def planning(root, protocol):
    budget = read(root/'artifacts/budget.json')
    plan = old.planning(root, budget, protocol)
    if budget['spent'] + budget['reserved'] + 8 > 200 + 1e-8:
        raise ValueError('Full eight dollar credit headroom required')
    return dict(budget=budget, **plan)


def verify_ready(root, target, rid, spec, own, replica, supplement, protocol):
    validate(spec, own, rid, target); sources(root, spec)
    grouped.identity(root, target, rid)
    if spec['runner_sha256'] != replica['runner_sha256']:
        raise ValueError('Replica provenance changed')
    prereq = prerequisites(root, target, rid, spec, protocol)
    if any((target/p).exists() for p in (ATTEMPT, MARKER, REPORT_DIR, old.ATTEMPT)):
        raise ValueError('Previous supplement attempt/report requires separate review')
    if old.cache_inventory(target, protocol) != spec['cache_sha256_or_absent']:
        raise ValueError('Supplement cache changed since admission')
    prior = grouped.capture_prior(root, target, rid, supplement, protocol, target/ATTEMPT/'inference')
    cost = planning(root, protocol)
    if old.cache_inventory(target, protocol) != spec['cache_sha256_or_absent']:
        raise ValueError('Supplement cache changed during replay')
    return dict(prerequisites=prereq, prior=prior, cost=cost,
                **common.current_scheduler(root, target, rid, spec, own))


def finish_report(root, target, rid, supplement, protocol, analysis, journal, staging):
    """Frozen analysis in private staging, six exclusive files then marker last."""
    staging = files.own_path(target, staging)
    if any((target/p).exists() for p in (REPORT_DIR, MARKER)) or staging.exists():
        raise ValueError('Existing report/partial publication requires review')
    verify(root, rid)
    protected = grouped.sources(target, supplement, protocol)
    before = {p: grouped.prefix(target/p) for p in grouped.layout(target, protocol)}
    ends = grouped.model_completions(journal, protocol)
    if ends['status'] != 'complete' or not all(c['complete'] for c in grouped.component.inventory(target, rid, protocol)):
        raise ValueError('Full raw supplement and model ends required')
    staging.mkdir(exist_ok=False)
    load = bind(analysis.load, PROJECT=target)
    bind(supplement.analyze, PROJECT=staging, load=load, verify=supplement.verify)()
    out = staging/REPORT_DIR
    if {p.name for p in out.iterdir()} != set(REPORT_NAMES):
        raise ValueError('Unexpected frozen supplement outputs')
    verify(root, rid)
    if (protected != grouped.sources(target, supplement, protocol)
            or before != {p: grouped.prefix(target/p) for p in grouped.layout(target, protocol)}
            or ends != grouped.model_completions(journal, protocol)):
        raise ValueError('Report sources changed')
    output = target/REPORT_DIR
    output.parent.mkdir(parents=True, exist_ok=True); files.sync_directory(output.parent.parent)
    output.mkdir(exist_ok=False); files.sync_directory(output.parent)
    for name in REPORT_NAMES:
        files.durable_write(output/name, (out/name).read_bytes())
    files.sync_directory(output)
    report = {REPORT_DIR+'/'+n: sha(output/n) for n in REPORT_NAMES}
    marker = dict(status='complete', run_id=rid, cells=7, report_sha256=report,
                  model_journal_sha256=ends['sha256'], full_expanded_study_complete=False)
    files.durable_write(target/MARKER, (protocol.canonical(marker)+'\n').encode())
    return {**report, MARKER: sha(target/MARKER)}


def execute_bound(root, target, rid, admission_path, spec, own, replica,
                  inference, supplement, protocol, analysis, orchestration):
    validate(spec, own, rid, target)
    with (root/'artifacts/operations/repetition_writer.lock').open('a') as shared, \
            (target/'artifacts/parallel_writer.lock').open('a') as exclusive:
        fcntl.flock(shared, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        admission_path = files.own_path(root, admission_path)
        if read(admission_path) != spec:
            raise ValueError('Admission changed')
        admission_sha = sha(admission_path)
        checked = verify_ready(root, target, rid, spec, own, replica, supplement, protocol)
        if sha(admission_path) != admission_sha:
            raise ValueError('Admission changed during review')
        attempt = target/ATTEMPT; attempt.mkdir(exist_ok=False); files.sync_directory(attempt.parent)
        journal = attempt/'inference'; prior_path = attempt/'prior.json'
        if grouped.capture_prior(root, target, rid, supplement, protocol, journal) != checked['prior']:
            raise ValueError('Prior changed before durable receipt')
        files.durable_write(prior_path, (protocol.canonical(checked['prior'])+'\n').encode())
        prior_sha = sha(prior_path)
        if read(prior_path) != checked['prior']:
            raise ValueError('Prior receipt readback mismatch')
        files.durable_write(attempt/'cost.json', (protocol.canonical(checked['cost'])+'\n').encode())
        history = target/'artifacts/stage_execution.jsonl'; old_sha = sha(history)
        def log(status, **extra):
            row = dict(utc=datetime.now(timezone.utc).isoformat(), stage='supplement', status=status,
                run_id=rid, slurm_job_id=own, admission_sha256=admission_sha,
                prior_stage_journal_sha256=old_sha, prior_cache_sha256=prior_sha,
                launcher_sha256=sha(root/SELF), parallel_policy_sha256=spec['parallel_policy_sha256'],
                budget_authority_sha256=APPROVAL_SHA, historical_unjournaled_outcomes_resolved=False, **extra)
            durable_append(attempt/'execution.jsonl', row, protocol.canonical)
            durable_append(history, row, protocol.canonical)
        log('started', verification={k:v for k,v in checked.items() if k not in ('prior','cost')})
        started_sha = sha(history); credit = None
        def gates():
            if sha(admission_path) != admission_sha:
                raise ValueError('Admission changed')
            sources(root, spec); verify(root, rid)
            prerequisites(root, target, rid, spec, protocol, started_sha)
            common.current_scheduler(root, target, rid, spec, own)
        def reconcile():
            return grouped.reconcile(root, target, rid, prior_path, prior_sha, journal, credit, supplement, protocol)
        try:
            gates()
            if grouped.capture_prior(root, target, rid, supplement, protocol, journal) != checked['prior']:
                raise ValueError('Prior cache changed before funding')
            planning(root, protocol)  # Fresh budget and cost sources immediately before funding.
            credit = Credit(root, f'fresh-supplement-{rid}-{own}', dict(
                admission_sha256=admission_sha, prior_sha256=prior_sha, run_id=rid,
                stage='supplement', journal_dir=str(journal.relative_to(target))))
            result = grouped.private_component(credit, prior_path, prior_sha)(
                root, target, rid, inference, supplement, protocol, journal)
            reconciled = reconcile()
            files.durable_write(attempt/'credit.json', (protocol.canonical(
                dict(raw=reconciled, credit=credit.summary()))+'\n').encode())
            if not result['inference_complete'] or not reconciled['inference_complete']:
                raise RuntimeError('Supplement failed after all model tasks drained')
            gates()
            report = finish_report(root, target, rid, supplement, protocol, analysis, journal, attempt/'report_staging')
            bind(orchestration.stamp, PROJECT=target)('supplement_billiger50', 'complete_automated',
                '7 raw-verified cells; manuscript integration pending')
            log('complete', report_sha256=report, credit_reconciliation=reconciled)
        except BaseException as exc:
            if credit is not None and not credit.closed:
                if isinstance(exc, OSError): credit.journal_io_failed = True
                try: reconcile()
                except BaseException as error:
                    log('credit_retained', exception_type=type(error).__name__, credit=credit.summary())
            log('failed', exception_type=type(exc).__name__)
            raise
    return 0


def execute(root, rid, admission_path):
    root = root.resolve(); target = replica_path(root, rid)
    spec = read(admission_path); own = os.environ.get('SLURM_JOB_ID')
    validate(spec, own, rid, target)
    if any(m in sys.modules for m in ('run','protocol','billiger50','analyze','orchestrate')):
        raise ValueError('Fresh Python process required')
    replica = verify(root, rid); sys.path.insert(0, str(target/'src'))
    import run, protocol, billiger50, analyze, orchestrate
    if any(m.PROJECT != target for m in (run,billiger50,analyze,orchestrate)):
        raise ValueError('Mixed replica modules')
    orchestrate.verify()
    return execute_bound(root,target,rid,admission_path,spec,own,replica,run,billiger50,protocol,analyze,orchestrate)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--run-id',type=int,choices=(2,3),required=True); ap.add_argument('--admission',type=Path,required=True)
    args = ap.parse_args(); raise SystemExit(execute(args.root,args.run_id,args.admission.resolve()))
