"""Bounded first fresh rules stage, privately composed from verified helpers.

No submission/retry. Only canonical successful prerequisite histories are
understood here. Current multi-recovery histories deliberately remain rejected;
their separate full-history adapter requires actual verified Core success.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import types

from authorized_recovery_200 import APPROVAL, APPROVAL_SHA, authority
from drained_inference import durable_append
import fresh_rules_drained as component
import fresh_rules_cost_plan as costs
import fresh_rules_grouped as grouped
import fresh_schema_selection as files
from fresh_schema_selected_cost import budget_review, require_headroom
from leased_budget import Credit
from repeated_runs import replica_path, sha, verify
from rules_drained import bind
import run_fresh_confirmation_drained as common
import fresh_confirmation_test_drained as confirmation_report
import fresh_schema_test_drained as schema_report

SELF = 'artifacts/operations/run_fresh_rules_drained.py'
ATTEMPT = 'artifacts/fresh_rules_drained_v1'
COST = 'artifacts/verification/fresh_rules_cost_scenario_20261016.json'
COST_SHA = 'feb8fb794c8393894b1640890647971845f71b40a440ec26d9816ce3335f5f99'
PROOF = 'artifacts/verification/fresh_rules_cost_verification_20261016.json'
PROOF_SHA = 'e9450b7f1c06d653354ddfcc1cb8e88c2b5771df52835542ee08a7143e0fc179'
REQUIRED = common.REQUIRED | {SELF, APPROVAL} | {f'artifacts/operations/{n}.py' for n in (
    'run_fresh_confirmation_drained', 'fresh_rules_drained', 'fresh_rules_cost_plan',
    'fresh_rules_grouped', 'rules_drained', 'rules_remaining_cost_plan',
    'fixed_extension_cost_plan', 'fresh_schema_selection', 'fresh_schema_selected_cost',
    'fresh_schema_test_drained', 'grouped_inference', 'leased_budget')}
read = common.read


def rows(path):
    grouped.prefix(path)
    return common.rows(path)


def validate(spec, own, rid, target):
    expected = dict(admitted=True, stage='rules', run_id=rid, slurm_job_id=own,
        replica=str(target), budget_cap_usd=200, budget_authority_sha256=APPROVAL_SHA,
        max_invocations_per_model_per_phase=1, historical_unknown_outcomes_disclosed=True,
        accounting_transport='bounded_process_credit', journal_transport='group_commit_v1',
        credit_chunk_usd=8, checkpoint_reservations=512,
        history_adapter='canonical_success_only_v1')
    if (not own or type(rid) is not int or rid not in (2, 3)
            or any(spec.get(k) != v for k, v in expected.items())):
        raise ValueError('Actual reviewed fresh rules admission required')
    if not REQUIRED <= spec.get('source_sha256', {}).keys():
        raise ValueError('Executable provenance incomplete')


def report_paths(target, stage):
    if stage == 'smoke':
        return set()
    if stage == 'core':
        return common.CORE_OUTPUTS
    if stage in ('confirmation', 'schema'):
        module = confirmation_report if stage == 'confirmation' else schema_report
        return {module.MARKER, *(module.REPORT_DIR+'/'+n for n in module.REPORT_NAMES)}
    folder = 'artifacts/results/'+('billiger50' if stage == 'supplement' else 'expanded_'+stage)
    found = {str(p.relative_to(target)) for p in (target/folder).iterdir() if p.is_file()}
    if not found:
        raise ValueError('Empty prerequisite reports')
    if stage in ('calibration', 'stability'):
        found.add(f'artifacts/expanded_{stage}_complete.json')
    return found


def prerequisites(root, target, rid, spec, protocol, started_sha=None):
    """Exhaust the canonical history, including smoke; never skip recovery rows."""
    path = target/'artifacts/stage_execution.jsonl'
    if sha(path) != (started_sha or spec['prerequisite_stage_journal_sha256']):
        raise ValueError('Prerequisite stage history changed')
    raw_lines = path.read_bytes().splitlines(keepends=True)
    history = rows(path)
    if started_sha:
        own = history.pop(); raw_lines.pop()
        if any(own.get(k) != v for k,v in dict(stage='rules', status='started',
                run_id=rid, slurm_job_id=spec['slurm_job_id']).items()):
            raise ValueError('Wrong own active stage')
    names = [r['stage'] for r in spec['prerequisites']]
    allowed = [prefix+tail for prefix in (
        ['smoke','core','supplement','screen'],
        ['smoke','core','supplement','screen','calibration'],
        ['smoke','core','supplement','screen','calibration','stability'])
        for tail in (['confirmation','schema'], ['schema','confirmation'])]
    if names not in allowed or len(history) != 2*len(names):
        raise ValueError('Unreviewed entire prerequisite history; recovery adapter is not admitted')
    jobs = {spec['slurm_job_id']}; accounts = {}; protected = {}
    for i, info in enumerate(spec['prerequisites']):
        stage, job = info['stage'], str(info['slurm_job_id'])
        if job in jobs:
            raise ValueError('Distinct prerequisite jobs required')
        jobs.add(job)
        pair = history[2*i:2*i+2]
        if ([r.get('status') for r in pair] != ['started','complete']
                or any(r.get('stage') != stage or type(r.get('run_id')) is not int
                    or r['run_id'] != rid or str(r.get('slurm_job_id')) != job for r in pair)
                or protocol.digest(pair[-1]) != info['stage_end_sha256']):
            raise ValueError('Complete exact prerequisite stage provenance required')
        admission = files.own_path(root, info['admission'])
        if (sha(admission) != info['admission_sha256']
                or any(r.get('admission_sha256') != info['admission_sha256'] for r in pair)):
            raise ValueError('Prerequisite admission provenance changed')
        obj = read(admission)
        if any(obj.get(k) != v for k,v in dict(admitted=True,stage=stage,run_id=rid,slurm_job_id=job).items()):
            raise ValueError('Prerequisite admission identity mismatch')
        # Fresh extensions must bind the complete prior byte prefix, not just
        # their immediate predecessor. Initial smoke/core have older schemas.
        if stage not in ('smoke','core'):
            digest = hashlib.sha256(b''.join(raw_lines[:2*i])).hexdigest()
            if (obj.get('prerequisite_stage_journal_sha256') != digest
                    or any(r.get('prior_stage_journal_sha256') != digest for r in pair)):
                raise ValueError('Prerequisite full history prefix not bound')
        reports = info['report_sha256']
        if set(reports) != report_paths(target, stage):
            raise ValueError('Incomplete prerequisite report inventory')
        if stage != 'smoke' and pair[-1].get('report_sha256') != reports:
            raise ValueError('Prerequisite stage report provenance mismatch')
        for rel,h in reports.items():
            if sha(files.own_path(target,rel)) != h:
                raise ValueError('Prerequisite report changed')
            protected[rel] = h
        if stage in ('confirmation','schema','calibration','stability'):
            status = 'complete_with_explicit_unavailable_fits' if stage == 'calibration' else 'complete'
            if read(target/f'artifacts/expanded_{stage}_complete.json').get('status') != status:
                raise ValueError('Prerequisite completion marker invalid')
        accounts[job] = common.accounting(job)  # Every main/batch/extern/extra step.
    expected = dict(smoke='complete',AP1_baseline='complete',AP2_development='complete',
        AP2_confirmation='complete',AP1_timing='complete',AP3_AP4_analysis='complete_automated',
        supplement_billiger50='complete_automated',expanded_E2_E3_screen='complete')
    progress = read(target/'artifacts/progress.json')
    if (sha(target/'artifacts/progress.json') != spec['progress_sha256']
            or set(progress) != set(expected) or any(progress[k]['status'] != v for k,v in expected.items())):
        raise ValueError('Full prerequisite progress required')
    component.prerequisites(target)
    return dict(accounting_utc=accounts, protected_reports_sha256=protected,
        history_adapter='canonical_success_only_v1', current_recovery_histories_admitted=False)


def snapshot(target, rules, protocol):
    _, _, cells, protected = component.prepare(target,rules,protocol)
    controls = [dict(c,variant='B0',namespace='quality') for c in cells if c['variant']=='R_rules']
    grouped.layout(target,cells+controls)
    return dict(protected_sha256=protected, journals_sha256_or_absent=component.hashes(target,cells+controls))


def cost_sources(root, protocol):
    if sha(root/COST) != COST_SHA or sha(root/PROOF) != PROOF_SHA:
        raise ValueError('Verified rules cost provenance changed')
    cost, proof = read(root/COST), read(root/PROOF)
    if proof['scenario_sha256'] != COST_SHA or cost['quote_or_hard_bound'] is not False:
        raise ValueError('Invalid historical rules cost provenance')
    expected = {f'runs/quality/{m}/{v}/{b}/prompt{s}' for m in protocol.MODELS
        for v in ('B0','X_P_domain_0') for b in protocol.BENCHMARKS for s in component.SUFFIXES}
    for p in expected:
        if cost['journal_sha256_or_absent'].get(p,'missing') != files.sha(files.own_path(root,p)):
            raise ValueError('Run1 development cost journal changed')
    protected = {p:h for p,h in cost['protected_sha256'].items()
        if p.startswith(('src/','data/','configs/learned_rules','configs/rule_agent_training','configs/expanded_rules','artifacts/astra_agent/'))}
    if not protected or not {f'data/{b}/prompt.json' for b in protocol.BENCHMARKS} <= protected.keys():
        raise ValueError('Historical cost source protection incomplete')
    component.unchanged(root,protected)
    return cost['sources']


def planning(root,target,rid,rules,protocol):
    # Recompute the actual own-run manual discount from raw records. The old
    # preparation's empty cache or hypothetical future confirmation is unused.
    inv = costs.inventory(root,target,rid,rules,protocol)
    plan = costs.price(cost_sources(root,protocol),inv['cells'],protocol.MODELS,protocol.BENCHMARKS)
    checked = budget_review(plan['totals'],read(root/'artifacts/budget.json'))
    if not checked['scenario_fits_current_headroom']:
        raise ValueError('Full actual rules/manual scenario exceeds current headroom')
    return dict(**plan,**plan['totals'],**checked,scenario_sha256=COST_SHA,
        verification_sha256=PROOF_SHA, own_cache=inv, quote_or_hard_bound=False,
        hypothetical_confirmation_credit=False,crossrun_answer_reuse=False)


def verify_ready(root,target,rid,spec,own,replica,rules,protocol):
    validate(spec,own,rid,target); authority(root); grouped.identity(root,target,rid)
    if spec['runner_sha256'] != replica['runner_sha256']:
        raise ValueError('Replica provenance changed')
    for p,h in spec['source_sha256'].items():
        if sha(files.own_path(root,p)) != h:
            raise ValueError('Operational source drift: '+p)
    prereq = prerequisites(root,target,rid,spec,protocol)
    component.untouched(target,protocol)
    before = snapshot(target,rules,protocol)
    if before != spec['cache_inventory']:
        raise ValueError('Source/cache changed since admission')
    prior = grouped.capture_prior(root,target,rid,rules,protocol)
    cost = planning(root,target,rid,rules,protocol)
    if snapshot(target,rules,protocol) != before:
        raise ValueError('Cache changed during admission replay')
    return dict(prerequisites=prereq,prior=prior,cost=cost,
        **common.current_scheduler(root,target,rid,spec,own))


def private_component(credit,history,started_sha):
    def untouched(target,protocol):
        if sha(history) != started_sha:
            raise ValueError('Own stage history changed')
        if any(r.get('stage')=='rules' for r in rows(history)[:-1]):
            raise ValueError('Prior rules stage')
        if any((target/p).exists() for p in (grouped.MARKER,grouped.REPORT_DIR,'artifacts/rules_cost_forecast.json')):
            raise ValueError('Existing rules output')
        if list((target/'runs/rules').rglob('*.jsonl')):
            raise ValueError('First rules attempt requires absent rule journals')
    local_component = types.SimpleNamespace(**{**vars(component),
        'run_rules':bind(component.run_rules,untouched=untouched)})
    return bind(grouped.private_component,component=local_component)(credit)


def execute_bound(root,target,rid,admission_path,spec,own,replica,inference,rules,protocol,analysis):
    validate(spec,own,rid,target)
    with (root/'artifacts/operations/repetition_writer.lock').open('a') as shared, \
            (target/'artifacts/parallel_writer.lock').open('a') as exclusive:
        fcntl.flock(shared,fcntl.LOCK_SH|fcntl.LOCK_NB)
        fcntl.flock(exclusive,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (target/ATTEMPT).exists():
            raise ValueError('Prior attempt requires separate terminal review')
        admission_path = files.own_path(root,admission_path)
        if read(admission_path) != spec:
            raise ValueError('Admission changed')
        admission_sha = sha(admission_path)
        checked = verify_ready(root,target,rid,spec,own,replica,rules,protocol)
        if sha(admission_path) != admission_sha:
            raise ValueError('Admission changed during review')
        attempt = target/ATTEMPT; attempt.mkdir(exist_ok=False); files.sync_directory(attempt.parent)
        # Receipt is exclusive and durable BEFORE funding. Replay once more
        # after current gates and compare to prevent a stale manual discount.
        prior_path = attempt/'prior.json'
        if grouped.capture_prior(root,target,rid,rules,protocol) != checked['prior']:
            raise ValueError('Manual/control prior changed before durable receipt')
        files.durable_write(prior_path,(protocol.canonical(checked['prior'])+'\n').encode())
        prior_sha = sha(prior_path)
        files.durable_write(attempt/'cost.json',(protocol.canonical(checked['cost'])+'\n').encode())
        history = target/'artifacts/stage_execution.jsonl';old_sha = sha(history)
        def log(status,**extra):
            row = dict(utc=datetime.now(timezone.utc).isoformat(),stage='rules',status=status,
                run_id=rid,slurm_job_id=own,admission_sha256=admission_sha,
                prior_stage_journal_sha256=old_sha,prior_cache_sha256=prior_sha,
                launcher_sha256=sha(root/SELF),parallel_policy_sha256=spec['parallel_policy_sha256'],
                budget_authority_sha256=APPROVAL_SHA,historical_unjournaled_outcomes_resolved=False,**extra)
            durable_append(attempt/'execution.jsonl',row,protocol.canonical)
            durable_append(history,row,protocol.canonical)
        log('started',verification={k:v for k,v in checked.items() if k not in ('prior','cost')})
        started_sha = sha(history); credit = None; journals = attempt/'inference'
        def reconcile():
            return grouped.reconcile(root,target,rid,prior_path,prior_sha,journals,credit,rules,protocol)
        try:
            if sha(admission_path) != admission_sha:
                raise ValueError('Admission changed before funding')
            prerequisites(root,target,rid,spec,protocol,started_sha)
            common.current_scheduler(root,target,rid,spec,own)
            for p,h in spec['source_sha256'].items():
                if sha(files.own_path(root,p)) != h:
                    raise ValueError('Operational source changed before funding')
            cost_sources(root,protocol)
            grouped.read_prior(root,target,rid,prior_path,prior_sha,rules,protocol)
            # Exact prefix equality is needed before any new attempts exist.
            if any(grouped.prefix(target/p) != old for p,old in checked['prior']['journals'].items()):
                raise ValueError('Prior cache changed before funding')
            require_headroom(root,checked['cost'])
            credit = Credit(root,f'fresh-rules-{rid}-{own}',dict(admission_sha256=admission_sha,
                prior_sha256=prior_sha,run_id=rid,cost_sha256=sha(attempt/'cost.json')))
            result = private_component(credit,history,started_sha)(root,target,rid,inference,rules,protocol,journals)
            reconciled = reconcile()
            files.durable_write(attempt/'credit.json',(protocol.canonical(dict(raw=reconciled,credit=credit.summary()))+'\n').encode())
            if not result['inference_complete']:
                raise RuntimeError('Rules failed after complete phase model drain')
            verify(root,rid)
            if sha(admission_path) != admission_sha:
                raise ValueError('Admission changed before report')
            prerequisites(root,target,rid,spec,protocol,started_sha)
            common.current_scheduler(root,target,rid,spec,own)
            for p,h in spec['source_sha256'].items():
                if sha(files.own_path(root,p)) != h:
                    raise ValueError('Operational source changed before report')
            report = grouped.finish_report(root,target,rid,rules,protocol,analysis,journals,attempt/'report_staging')
            log('complete',report_sha256=report,credit_reconciliation=reconciled)
        except BaseException as exc:
            if credit is not None and not credit.closed:
                if isinstance(exc,OSError):credit.journal_io_failed=True
                try:reconcile()
                except BaseException as error:
                    log('credit_retained',exception_type=type(error).__name__,credit=credit.summary())
            log('failed',exception_type=type(exc).__name__)
            raise
    return 0


def execute(root,rid,admission_path):
    root = root.resolve(); target = replica_path(root,rid)
    spec = read(admission_path); own = os.environ.get('SLURM_JOB_ID')
    validate(spec,own,rid,target)
    if any(m in sys.modules for m in ('run','protocol','expanded_rules','analyze','orchestrate')):
        raise ValueError('Fresh Python process required')
    replica = verify(root,rid); sys.path.insert(0,str(target/'src'))
    import run, protocol, expanded_rules, analyze, orchestrate
    if any(m.PROJECT != target for m in (run,expanded_rules,analyze,orchestrate)):
        raise ValueError('Mixed replica modules')
    orchestrate.verify()
    return execute_bound(root,target,rid,admission_path,spec,own,replica,run,expanded_rules,protocol,analyze)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--run-id',type=int,choices=(2,3),required=True); ap.add_argument('--admission',type=Path,required=True)
    args = ap.parse_args(); raise SystemExit(execute(args.root,args.run_id,args.admission.resolve()))
