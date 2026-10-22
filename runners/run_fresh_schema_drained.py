"""Bounded first fresh schema stage with separately priced development and test.

No submission/retry. Global SH and replica EX remain held through final marker.
Existing complete components are privately bound to grouped writes and credits;
credits close only after exact durable-attempt reconciliation. A stopped freeze
needs separate recovery. Current multi-resume Core histories are NOT admitted by
this wrapper: their prerequisite adapter requires a later verified Core success.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import types

from authorized_recovery_200 import APPROVAL, APPROVAL_SHA, authority
from drained_inference import durable_append
import fresh_schema_development_drained as development
import fresh_schema_selection as freeze
import fresh_schema_test_drained as tests
import fresh_schema_selected_cost as selected_cost
from grouped_inference import build as grouped_build
from grouped_resume_20261015 import new_attempt_costs
from leased_budget import Credit
from repeated_runs import replica_path, sha, verify
import run_fresh_confirmation_drained as common
from run_fresh_supplement_drained import core_history, bind

SELF = 'artifacts/operations/run_fresh_schema_drained.py'
ATTEMPT = 'artifacts/fresh_schema_drained_v1'
COST = 'artifacts/verification/fresh_schema_cost_scenarios_20261015.json'
COST_SHA = 'a321076856c9738322534804f691efeb8c02a62db4577934b4a2d41af18909b9'
PROOF = 'artifacts/verification/fresh_schema_cost_verification_20261015.json'
PROOF_SHA = 'a83601f962deabdcd9b804698b8616204ef2ba0a8fba90886d06f957bcba38e6'
REQUIRED = common.REQUIRED | {SELF, APPROVAL}
REQUIRED |= {f'artifacts/operations/{n}.py' for n in (
    'run_fresh_confirmation_drained', 'fresh_schema_development_drained', 'fresh_schema_selection',
    'fresh_schema_test_drained', 'fresh_schema_selected_cost', 'fresh_schema_cost_plan',
    'grouped_inference', 'grouped_resume_20261015', 'leased_budget', 'grouped_transition',
    'fast_transition_audit', 'schema_drained', 'schema_run_export', 'diagnose_schema_terminal')}
read, rows = common.read, common.rows


def validate(spec, own, run_id, target):
    required = dict(admitted=True, stage='schema', run_id=run_id, slurm_job_id=own,
        replica=str(target), budget_cap_usd=200, budget_authority_sha256=APPROVAL_SHA,
        max_invocations_per_model_per_phase=1, historical_unknown_outcomes_disclosed=True,
        two_phase_budget_gate=True, accounting_transport='bounded_process_credit',
        journal_transport='group_commit_v1', credit_chunk_usd=8, checkpoint_reservations=512)
    if (not own or type(run_id) is not int or run_id not in (2, 3)
            or any(spec.get(k) != v for k, v in required.items())):
        raise ValueError('Actual reviewed two-phase schema admission required')
    if not REQUIRED <= spec.get('source_sha256', {}).keys():
        raise ValueError('Executable provenance incomplete')


def cache_inventory(target, schema):
    paths = {p for _, _, _, _, input_path, journals in freeze.cell_paths(target, schema)
             for p in (input_path, *journals)}
    paths |= tests.paths(target, [(m, v, b) for m in schema.MODELS
        for v in [*schema.CONDITIONS, 'B0'] for b in [*schema.BENCHMARKS, 'billiger-de-50un']])
    for ns in ('schema', 'schema_pilot'):
        if not set((target/'runs'/ns).rglob('*.jsonl')) <= paths:
            raise ValueError('Unreviewed schema journal')
    return {str(freeze.own_path(target, p).relative_to(target)): freeze.sha(p) for p in sorted(paths)}


def planning(root, budget, schema):
    if sha(root/COST) != COST_SHA or sha(root/PROOF) != PROOF_SHA:
        raise ValueError('Verified pilot/development cost provenance changed')
    cost, proof = read(root/COST), read(root/PROOF)
    if proof['scenario_sha256'] != COST_SHA or cost['quote_or_hard_bound'] is not False:
        raise ValueError('Historical planning provenance invalid')
    protected = cost['protected_sha256_or_absent']
    paths = {str(p.relative_to(root)) for _, _, _, _, input_path, journals in freeze.cell_paths(root, schema)
             for p in (input_path, *journals)}
    if any(protected.get(p, 'missing') != freeze.sha(root/p) for p in paths):
        raise ValueError('Run1 pilot/development cost source changed')
    stages = cost['pilot_and_development']
    if (stages['schema_pilot']['cells'] != 70 or stages['schema_pilot']['prospective_responses'] != 140
            or stages['schema']['cells'] != 280 or stages['schema']['prospective_responses'] != 56000):
        raise ValueError('Wrong complete pilot/development matrix')
    total = {k: math.fsum(v[k] for v in stages.values())
             for k in ('known_proxy_usd', 'historical_unknown_scenario_usd')}
    if total != cost['pilot_and_development_total']:
        raise ValueError('Pilot/development totals changed')
    checked = selected_cost.budget_review(total, budget)
    if not checked['scenario_fits_current_headroom']:
        raise ValueError('Full pilot/development scenario exceeds current headroom')
    return dict(scenario_sha256=COST_SHA, verification_sha256=PROOF_SHA,
        pilot_and_development=stages, **total, **checked, test_admission_granted=False,
        quote_or_hard_bound=False, baseline_or_crossrun_credit=False)


def prerequisites(root, target, rid, spec, history, schema):
    # Deliberately narrow history. Do not generalize/ignore fast or grouped
    # recovery events before a separate full terminal Core proof exists.
    if spec.get('previous_completed_stages', []) != []:
        raise ValueError('Intervening stages need a separately reviewed history adapter')
    core, supplement = str(spec['core_job_id']), str(spec['supplement_job_id'])
    if len({core, supplement, spec['slurm_job_id']}) != 3:
        raise ValueError('Distinct prerequisite jobs required')
    last = history[-2:]
    if (len(last) != 2 or [r.get('status') for r in last] != ['started', 'complete']
            or any(r.get('stage') != 'supplement' or r.get('run_id') != rid
                   or str(r.get('slurm_job_id')) != supplement for r in last)):
        raise ValueError('Actual supplement completion required')
    core_history(history[:-2], rid, core)
    end = history[-3]
    for name, event, job in [('core', end, core), ('supplement', last[-1], supplement)]:
        admission = (root/spec[name+'_admission']).resolve()
        if (not admission.is_relative_to(root) or sha(admission) != spec[name+'_admission_sha256']
                or event.get('admission_sha256') != spec[name+'_admission_sha256']
                or schema.protocol.digest(event) != spec[name+'_stage_end_sha256']):
            raise ValueError('Prerequisite completion/admission provenance changed')
        obj = read(admission)
        if any(obj.get(k) != v for k, v in dict(admitted=True, stage=name, run_id=rid, slurm_job_id=job).items()):
            raise ValueError('Prerequisite admission identity mismatch')
        reports = spec[name+'_report_sha256']
        if name == 'core':
            if set(reports) != common.CORE_OUTPUTS:
                raise ValueError('Incomplete core reports')
        else:
            observed = {str(p.relative_to(target)):sha(p) for p in (target/'artifacts/results/billiger50').iterdir() if p.is_file()}
            if not observed or observed != reports or event.get('report_sha256') != reports:
                raise ValueError('Supplement report provenance changed')
        if any(not (target/p).resolve().is_relative_to(target) or sha(target/p) != h for p,h in reports.items()):
            raise ValueError('Prerequisite report changed')
    required = dict(smoke='complete', AP1_baseline='complete', AP2_development='complete',
        AP2_confirmation='complete', AP1_timing='complete', AP3_AP4_analysis='complete_automated',
        supplement_billiger50='complete_automated')
    progress = read(target/'artifacts/progress.json')
    if set(progress) != set(required) or any(progress[k]['status'] != v for k,v in required.items()):
        raise ValueError('Full core/supplement progress required; no artificial screen prerequisite')
    if sha(target/'artifacts/progress.json') != spec['progress_sha256']:
        raise ValueError('Prerequisite progress changed')
    controls = {(m,'B0',b) for m in schema.MODELS for b in [*schema.BENCHMARKS,'billiger-de-50un']}
    if not all(c['complete'] for c in tests.inventory(target,rid,schema,controls)):
        raise ValueError('Full own B0 test controls required')
    return {j: common.accounting(j) for j in (core,supplement)}


def verify_ready(root, target, rid, spec, own, replica, schema, started_sha=None):
    validate(spec,own,rid,target); authority(root)
    if spec.get('runner_sha256') != replica['runner_sha256']:
        raise ValueError('Replica provenance changed')
    for p,h in spec['source_sha256'].items():
        path=(root/p).resolve()
        if not path.is_relative_to(root) or sha(path)!=h:
            raise ValueError('Operational source drift: '+p)
    history_path=target/'artifacts/stage_execution.jsonl'
    if sha(history_path)!=(started_sha or spec['prerequisite_stage_journal_sha256']):
        raise ValueError('Stage journal changed since reviewed boundary')
    history=rows(history_path)
    if started_sha:
        start=history.pop()
        if any(start.get(k)!=v for k,v in dict(stage='schema',status='started',run_id=rid,slurm_job_id=own).items()):
            raise ValueError('Wrong active schema stage')
    accounts=prerequisites(root,target,rid,spec,history,schema)
    freeze.identity(root,target,rid,schema)
    if started_sha is None:
        if cache_inventory(target,schema)!=spec['cache_sha256_or_absent']:
            raise ValueError('Schema cache changed since admission')
        if any(any((target/'runs'/ns).rglob('*')) for ns in ('schema','schema_pilot')):
            raise ValueError('Prior schema journals require separate terminal review')
        cost=planning(root,read(root/'artifacts/budget.json'),schema)
    else:
        cost=None  # Own full replay and actual selected costs checked next.
    return dict(cost=cost,prerequisite_accounting_utc=accounts,
        **common.current_scheduler(root,target,rid,spec,own))


def journal_snapshot(target):
    return {str(p.relative_to(target)):{'bytes':p.stat().st_size,'sha256':sha(p)}
        for ns in ('schema','schema_pilot') for p in (target/'runs'/ns).rglob('*.jsonl')}


def reconcile(credit, target, before):
    if getattr(credit,'journal_io_failed',False):
        raise ValueError('Journal durability failure; keep credit')
    for rel, prior in before.items():
        path=target/rel
        with path.open('rb') as stream:
            if hashlib.sha256(stream.read(prior['bytes'])).hexdigest()!=prior['sha256']:
                raise ValueError('Prior journal changed; keep credit')
    for ns in ('schema','schema_pilot'):
        for path in (target/'runs'/ns).rglob('*.operations.jsonl'):
            offset=before.get(str(path.relative_to(target)),{}).get('bytes',0)
            with path.open('rb') as stream:
                stream.seek(offset)
                if any(json.loads(line)['operational_stop']['outcome_unknown'] for line in stream):
                    raise ValueError('Unknown operational outcome; keep credit')
    observed=new_attempt_costs(target,before,{'schema','schema_pilot'})
    if (credit.outstanding or credit.attempts!=credit.settled or observed['attempts']!=credit.settled
            or observed['unknown']!=credit.unknown or not math.isclose(observed['charged'],credit.charged,rel_tol=0,abs_tol=1e-8)):
        raise ValueError('Unreconciled durable attempts or unknown outcome; keep credit')
    credit.checkpoint(close=True)
    return observed


def components(credit):
    # Rebind only private function globals, preserving the original frozen wire
    # and thread-local parser and without mutating any imported module.
    def builder(inference, rid):
        isolated=types.SimpleNamespace(**{**vars(inference),'account':credit.account,
            '_on_journal_failure':lambda:setattr(credit,'journal_io_failed',True)})
        return grouped_build(isolated,rid)
    build_schema=bind(development.build_schema,
        make_wrapper=lambda *_:types.SimpleNamespace(authorized_build=builder))
    dev=types.SimpleNamespace(**{**vars(development),'build_schema':build_schema})
    dev.run_development=bind(development.run_development,build_schema=build_schema)
    test=bind(tests.run_selected_tests,development=dev)
    return dev.run_development,test


def execute_bound(root,target,rid,admission_path,spec,own,replica,inference,schema,analysis):
    validate(spec,own,rid,target)
    with (root/'artifacts/operations/repetition_writer.lock').open('a') as shared, \
            (target/'artifacts/parallel_writer.lock').open('a') as exclusive:
        fcntl.flock(shared,fcntl.LOCK_SH|fcntl.LOCK_NB)
        fcntl.flock(exclusive,fcntl.LOCK_EX|fcntl.LOCK_NB)
        forbidden=(ATTEMPT,freeze.ATTEMPT,tests.ATTEMPT,tests.REPORT_ATTEMPT,tests.MARKER,tests.REPORT_DIR,
            *freeze.OUTPUTS,'artifacts/schema_pilot_status.json',
            'artifacts/schema_development_cost_forecast.json','artifacts/schema_confirmation_cost_forecast.json')
        if any((target/p).exists() for p in forbidden):
            raise ValueError('Prior schema attempt/output requires separate review')
        if read(admission_path)!=spec:
            raise ValueError('Admission differs from loaded specification')
        admission_sha=sha(admission_path)
        checked=verify_ready(root,target,rid,spec,own,replica,schema)
        if sha(admission_path)!=admission_sha:
            raise ValueError('Admission changed during initial review')
        attempt=target/ATTEMPT;attempt.mkdir(exist_ok=False);freeze.sync_directory(attempt.parent)
        history=target/'artifacts/stage_execution.jsonl';prior=sha(history)
        def log(status,**extra):
            row=dict(utc=datetime.now(timezone.utc).isoformat(),run_id=rid,stage='schema',status=status,
                slurm_job_id=own,prior_stage_journal_sha256=prior,admission_sha256=admission_sha,
                launcher_sha256=sha(root/SELF),parallel_policy_sha256=spec['parallel_policy_sha256'],
                budget_authority_sha256=APPROVAL_SHA,historical_unjournaled_outcomes_resolved=False,**extra)
            durable_append(attempt/'execution.jsonl',row,schema.protocol.canonical)
            durable_append(history,row,schema.protocol.canonical)
        log('started',verification=checked);started_sha=sha(history)
        phase='development';credit=None;before=None
        try:
            # Re-read immediately before funding; credit is a reservation, not a charge.
            selected_cost.require_headroom(root,checked['cost'])
            before=journal_snapshot(target)
            freeze.durable_write(attempt/'development_prior.json',(json.dumps(before)+'\n').encode())
            credit=Credit(root,f'fresh-schema-{rid}-{own}-development',{'admission_sha256':admission_sha,'run_id':rid})
            dev,_=components(credit)
            result=dev(root,target,rid,inference,schema,analysis,attempt/'development')
            reconciled=reconcile(credit,target,before)
            freeze.durable_write(attempt/'development_credit.json',(json.dumps(dict(raw=reconciled,credit=credit.summary()))+'\n').encode())
            if not result['development_complete']:
                raise RuntimeError('Pilot/development failed after submitted tasks drained')
            phase='freeze'
            freeze.freeze_development(root,target,rid,schema,analysis,attempt/'development/models.jsonl')
            phase='selected_test_gate'
            verify(root,rid)
            if sha(admission_path)!=admission_sha:
                raise ValueError('Admission changed during development')
            current=verify_ready(root,target,rid,spec,own,replica,schema,started_sha)
            priced=selected_cost.review(root,target,rid,schema,analysis)
            current.update(common.current_scheduler(root,target,rid,spec,own))
            headroom=selected_cost.require_headroom(root,priced)
            if sha(admission_path)!=admission_sha or sha(history)!=started_sha:
                raise ValueError('Admission or stage changed before test')
            receipt=dict(run_id=rid,slurm_job_id=own,admission_sha256=admission_sha,
                current_gate=current,selected_test_costs=priced,fresh_budget_gate=headroom)
            freeze.durable_write(attempt/'selected_test_gate.json',(json.dumps(receipt)+'\n').encode())
            phase='test';before=journal_snapshot(target)
            freeze.durable_write(attempt/'test_prior.json',(json.dumps(before)+'\n').encode())
            selected_cost.require_headroom(root,priced)
            credit=Credit(root,f'fresh-schema-{rid}-{own}-test',{'admission_sha256':admission_sha,'run_id':rid})
            _,test=components(credit)
            result=test(root,target,rid,inference,schema,analysis)
            reconciled=reconcile(credit,target,before)
            freeze.durable_write(attempt/'test_credit.json',(json.dumps(dict(raw=reconciled,credit=credit.summary()))+'\n').encode())
            if not result['test_complete']:
                raise RuntimeError('Selected test failed after all model tasks drained')
            phase='report';verify(root,rid)
            if sha(admission_path)!=admission_sha:
                raise ValueError('Admission changed during tests')
            report=tests.finish_report(root,target,rid,schema,analysis)
            log('complete',report_sha256=report,selection_receipt_sha256=sha(target/freeze.RECEIPT),
                selected_test_gate_sha256=sha(attempt/'selected_test_gate.json'))
        except BaseException as exc:
            if credit is not None and not credit.closed:
                try:
                    reconcile(credit,target,before)
                except BaseException as error:
                    log('credit_retained',exception_type=type(error).__name__,credit=credit.summary())
            log('failed',phase=phase,exception_type=type(exc).__name__,selection_preserved=(target/freeze.RECEIPT).exists())
            raise
    return 0


def execute(root,rid,admission_path):
    root=root.resolve();target=replica_path(root,rid);spec=read(admission_path);own=os.environ.get('SLURM_JOB_ID')
    validate(spec,own,rid,target)
    if any(m in sys.modules for m in ('run','protocol','expanded_schema','analyze','orchestrate','billiger50')):
        raise ValueError('Fresh Python process required')
    replica=verify(root,rid);sys.path.insert(0,str(target/'src'))
    import run, expanded_schema, analyze, orchestrate, billiger50
    if any(m.PROJECT!=target for m in (run,expanded_schema,analyze,orchestrate,billiger50)):
        raise ValueError('Mixed replica modules')
    orchestrate.verify();billiger50.verify()
    return execute_bound(root,target,rid,admission_path,spec,own,replica,run,expanded_schema,analyze)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run-id',type=int,choices=(2,3),required=True)
    parser.add_argument('--admission',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(execute(args.root,args.run_id,args.admission.resolve()))
