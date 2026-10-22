"""First fresh variants wrapper; no submission, retries or real admission.

The narrow prerequisite adapter intentionally rejects current multi-resume Core
histories. Adapt their entire history only after actual verified Core success.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import types

from authorized_recovery_200 import APPROVAL, APPROVAL_SHA, authority
from drained_inference import durable_append
import fresh_variants_drained as component
import fresh_variants_cost_plan as costs
import fresh_schema_selection as files
from fresh_schema_selected_cost import require_headroom, budget_review
from grouped_inference import build as grouped_build
from leased_budget import Credit
from repeated_runs import replica_path, sha, verify
import run_fresh_confirmation_drained as common
from run_fresh_supplement_drained import bind, core_history

SELF = 'artifacts/operations/run_fresh_variants_drained.py'
ATTEMPT = 'artifacts/fresh_variants_drained_v1'
REPORT_ATTEMPT = ATTEMPT+'/report_staging'
REPORT_DIR = 'artifacts/results/expanded_variants'
MARKER = 'artifacts/expanded_variants_complete.json'
REPORT_NAMES = ('metrics.json', 'paired_language.json', 'costs.json', 'provenance.json', 'metrics.csv')
COST = 'artifacts/verification/fresh_variants_cost_scenario_20261016.json'
COST_SHA = 'ec6d48f7a4ac14c5a0335ab19aeba961e8309badd1826e0e92bb6e8102200f31'
PROOF = 'artifacts/verification/fresh_variants_cost_verification_20261016.json'
PROOF_SHA = 'f6846a5d3f1e0e407e09a5ff27175f8eda959f3fbc064d3f8c96004d646a45db'
REQUIRED = common.REQUIRED | {SELF, APPROVAL} | {f'artifacts/operations/{n}.py' for n in (
    'run_fresh_confirmation_drained', 'fresh_variants_drained', 'fresh_variants_cost_plan',
    'fresh_schema_selection', 'fresh_schema_selected_cost', 'grouped_inference', 'leased_budget')}
read, rows = common.read, common.rows


def validate(spec, own, rid, target):
    expected = dict(admitted=True, stage='variants', run_id=rid, slurm_job_id=own, replica=str(target),
        budget_cap_usd=200, budget_authority_sha256=APPROVAL_SHA, max_invocations_per_model=1,
        historical_unknown_outcomes_disclosed=True, accounting_transport='bounded_process_credit',
        journal_transport='group_commit_v1', credit_chunk_usd=8, checkpoint_reservations=512)
    if (not own or type(rid) is not int or rid not in (2,3)
            or any(spec.get(k)!=v for k,v in expected.items())):
        raise ValueError('Actual reviewed fresh variants admission required')
    if not REQUIRED <= spec.get('source_sha256',{}).keys():
        raise ValueError('Executable provenance incomplete')


def paths(target, protocol, controls=False):
    return [target/'runs/quality'/m/'B0'/b/('test'+s) for m in protocol.MODELS
        for b in (component.CONTROLS if controls else component.VARIANTS) for s in component.SUFFIXES]


def snapshot(target, protocol):
    expected=set(paths(target,protocol)+paths(target,protocol,True))
    for p in expected:
        files.own_path(target,p)
    for m in protocol.MODELS:
        for b in [*component.CONTROLS,*component.VARIANTS]:
            if not set((target/'runs/quality'/m/'B0'/b).glob('test*.jsonl')) <= expected:
                raise ValueError('Unreviewed variant/control journal')
    expected |= {target/'data'/b/'test.json' for b in [*component.CONTROLS,*component.VARIANTS]}
    return {str(p.relative_to(target)):files.sha(files.own_path(target,p)) for p in sorted(expected)}


def planning(root,budget,protocol):
    if sha(root/COST)!=COST_SHA or sha(root/PROOF)!=PROOF_SHA:
        raise ValueError('Verified variants cost provenance changed')
    cost,proof=read(root/COST),read(root/PROOF)
    if (proof['scenario_sha256']!=COST_SHA or cost['quote_or_hard_bound'] is not False
            or cost['cells_per_run']!=35 or cost['responses_per_run']!=156387):
        raise ValueError('Wrong complete historical variant scenario')
    protected=cost['protected_sha256']
    sources={str(p.relative_to(root)) for p in paths(root,protocol)}
    sources|={f'data/{b}/test.json' for b in component.VARIANTS}
    sources|={costs.AUDIT,costs.PROOF}
    absent=set(cost['absent_source_journals'])
    for p in sources:
        if files.sha(root/p)!=(None if p in absent else protected.get(p,'missing')):
            raise ValueError('Run1 variant cost source changed: '+p)
    expected={(m,b) for m in protocol.MODELS for b in component.VARIANTS}
    if {(c['model'],c['benchmark']) for c in cost['sources']}!=expected or len(cost['sources'])!=35:
        raise ValueError('Historical cost matrix incomplete')
    total=cost['per_run_total'];checked=budget_review(total,budget)
    if not checked['scenario_fits_current_headroom']:
        raise ValueError('Full variants scenario exceeds current headroom')
    return dict(**total,**checked,scenario_sha256=COST_SHA,verification_sha256=PROOF_SHA,
        quote_or_hard_bound=False,crossrun_answer_reuse=False)


def prerequisites(root,target,rid,spec,history,protocol):
    if spec.get('previous_completed_stages',[])!=[]:
        raise ValueError('Intervening stages need separately reviewed full history adapter')
    core=str(spec['core_job_id'])
    if core==spec['slurm_job_id']:
        raise ValueError('Distinct prerequisite job required')
    core_history(history,rid,core)
    end=history[-1];admission=(root/spec['core_admission']).resolve()
    if (not admission.is_relative_to(root) or sha(admission)!=spec['core_admission_sha256']
            or end.get('admission_sha256')!=spec['core_admission_sha256']
            or protocol.digest(end)!=spec['core_stage_end_sha256']):
        raise ValueError('Core completion/admission provenance changed')
    if any(read(admission).get(k)!=v for k,v in dict(admitted=True,stage='core',run_id=rid,slurm_job_id=core).items()):
        raise ValueError('Core admission identity mismatch')
    reports=spec['core_report_sha256']
    if set(reports)!=common.CORE_OUTPUTS:
        raise ValueError('Incomplete core report provenance')
    for p,h in reports.items():
        if sha(files.own_path(target,p))!=h:
            raise ValueError('Core reports changed')
    progress=read(target/'artifacts/progress.json')
    required=dict(smoke='complete',AP1_baseline='complete',AP2_development='complete',
        AP2_confirmation='complete',AP1_timing='complete',AP3_AP4_analysis='complete_automated')
    if (set(progress)!=set(required) or any(progress[k]['status']!=v for k,v in required.items())
            or sha(target/'artifacts/progress.json')!=spec['progress_sha256']):
        raise ValueError('Full actual core progress required')
    if not all(c['complete'] for c in component.inventory(target,rid,protocol,controls=True)):
        raise ValueError('Own fourteen B0 controls incomplete')
    return common.accounting(core)


def verify_ready(root,target,rid,spec,own,replica,variants,protocol):
    validate(spec,own,rid,target);authority(root)
    if target!=replica_path(root,rid) or spec['runner_sha256']!=replica['runner_sha256']:
        raise ValueError('Replica provenance changed')
    for p,h in spec['source_sha256'].items():
        path=(root/p).resolve()
        if not path.is_relative_to(root) or sha(path)!=h:
            raise ValueError('Operational source drift: '+p)
    history=target/'artifacts/stage_execution.jsonl'
    if sha(history)!=spec['prerequisite_stage_journal_sha256']:
        raise ValueError('Stage history changed')
    accounting=prerequisites(root,target,rid,spec,rows(history),protocol)
    variants.verify();component.untouched(target,protocol)
    if snapshot(target,protocol)!=spec['cache_sha256_or_absent']:
        raise ValueError('Inputs or cache changed since admission')
    cost=planning(root,read(root/'artifacts/budget.json'),protocol)
    return dict(cost=cost,core_accounting_utc=accounting,
        **common.current_scheduler(root,target,rid,spec,own))


def private_component(credit,history,started_sha):
    def untouched(target,protocol):
        if sha(history)!=started_sha:
            raise ValueError('Own stage history changed')
        # The wrapper just wrote the one pinned start. Every earlier variants
        # event was rejected before locks/funding; no prior history is erased.
        if any(r.get('stage')=='variants' for r in rows(history)[:-1]):
            raise ValueError('Prior variants stage')
        for p in (MARKER,REPORT_DIR,'artifacts/expanded_variants_cost_forecast.json'):
            if (target/p).exists():raise ValueError('Existing variant output')
        for m in protocol.MODELS:
            for b in component.VARIANTS:
                if list((target/'runs/quality'/m/'B0'/b).glob('*.jsonl')):
                    raise ValueError('First variant attempt requires absent journals')
    def builder(inference,rid):
        isolated=types.SimpleNamespace(**{**vars(inference),'account':credit.account,
            '_on_journal_failure':lambda:setattr(credit,'journal_io_failed',True)})
        return grouped_build(isolated,rid)
    return bind(component.run_variants,untouched=untouched,
        make_wrapper=lambda *_:types.SimpleNamespace(authorized_build=builder))


def reconcile(credit,target,protocol):
    if getattr(credit,'journal_io_failed',False):
        raise ValueError('Journal durability failed; keep credit')
    model_completions(target,protocol,successful=False)
    known=[];unknown=attempts=0
    for p in paths(target,protocol):
        if not p.exists():continue
        for row in rows(p):
            if p.name.endswith('.operations.jsonl'):
                if row['operational_stop']['outcome_unknown']:
                    raise ValueError('Unknown operational outcome; keep credit')
            elif not p.name.endswith('.executions.jsonl'):
                for a in row['attempts']:
                    attempts+=1
                    if a['cost'] is None:unknown+=1
                    else:known.append(a['cost'])
    charged=math.fsum(known)+.05*unknown
    if (credit.outstanding or credit.attempts!=credit.settled or attempts!=credit.settled
            or unknown!=credit.unknown or not math.isclose(charged,credit.charged,rel_tol=0,abs_tol=1e-8)):
        raise ValueError('Unreconciled durable attempts; keep credit')
    credit.checkpoint(close=True)
    return dict(attempts=attempts,unknown=unknown,charged=charged)


def model_completions(target,protocol,successful=True):
    path=target/ATTEMPT/'inference/models.jsonl';events=rows(path)
    if (len(events)!=len(protocol.MODELS)+2 or events[0].get('event')!='start'
            or events[0].get('models')!=list(protocol.MODELS)
            or events[0].get('invocations_per_model')!=1
            or events[-1].get('event')!='end' or events[-1].get('status') not in ('complete','failed')
            or events[-1].get('models_drained')!=len(protocol.MODELS)):
        raise ValueError('Complete model ends required')
    middle=events[1:-1]
    if ({r.get('model') for r in middle}!=set(protocol.MODELS)
            or any(r.get('event')!='model_end' or r.get('status') not in ('complete','failed') for r in middle)
            or (successful and (events[-1]['status']!='complete' or any(r['status']!='complete' for r in middle)))):
        raise ValueError('Complete model ends required')
    return sha(path)


def finish_report(root,target,rid,variants,protocol,analysis):
    if any((target/p).exists() for p in (REPORT_DIR,MARKER,REPORT_ATTEMPT)):
        raise ValueError('Existing report/partial publication requires review')
    verify(root,rid);variants.verify()
    ends=model_completions(target,protocol)
    if not all(c['complete'] for c in component.inventory(target,rid,protocol)+component.inventory(target,rid,protocol,True)):
        raise ValueError('Complete variant and own control raw matrix required')
    before=snapshot(target,protocol)
    staging=target/REPORT_ATTEMPT;staging.mkdir(exist_ok=False)
    loader=bind(analysis.load,PROJECT=target)
    bound=types.SimpleNamespace(**{**vars(analysis),'load':loader})
    markers=[]
    def capture(path,value):
        if path!=staging/MARKER:raise ValueError('Unexpected marker')
        markers.append(value)
    report=bind(variants.report,PROJECT=staging,verify=variants.verify,analyze=bound,write_frozen=capture)
    report()
    expected=dict(status='complete',new_cells=35,reused_core_cells=14,language_comparisons=21,
        wdc_full_split_paired_tests=False,full_expanded_study_complete=False)
    out=staging/REPORT_DIR
    if markers!=[expected] or {p.name for p in out.iterdir()}!=set(REPORT_NAMES):
        raise ValueError('Frozen report output mismatch')
    verify(root,rid);variants.verify()
    if before!=snapshot(target,protocol) or ends!=model_completions(target,protocol):
        raise ValueError('Report sources changed')
    for name in REPORT_NAMES:files.durable_write(target/REPORT_DIR/name,(out/name).read_bytes())
    files.sync_directory(target/REPORT_DIR)
    files.durable_write(target/MARKER,(protocol.canonical(expected)+'\n').encode())
    return {p:sha(target/p) for p in [MARKER,*(REPORT_DIR+'/'+n for n in REPORT_NAMES)]}


def execute_bound(root,target,rid,admission_path,spec,own,replica,inference,variants,protocol,analysis):
    validate(spec,own,rid,target)
    with (root/'artifacts/operations/repetition_writer.lock').open('a') as shared, \
            (target/'artifacts/parallel_writer.lock').open('a') as exclusive:
        fcntl.flock(shared,fcntl.LOCK_SH|fcntl.LOCK_NB);fcntl.flock(exclusive,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (target/ATTEMPT).exists():raise ValueError('Prior attempt requires separate review')
        if read(admission_path)!=spec:raise ValueError('Admission changed')
        admission_sha=sha(admission_path)
        checked=verify_ready(root,target,rid,spec,own,replica,variants,protocol)
        if sha(admission_path)!=admission_sha:raise ValueError('Admission changed during review')
        attempt=target/ATTEMPT;attempt.mkdir(exist_ok=False);files.sync_directory(attempt.parent)
        history=target/'artifacts/stage_execution.jsonl';prior=sha(history)
        def log(status,**extra):
            row=dict(utc=datetime.now(timezone.utc).isoformat(),stage='variants',status=status,run_id=rid,
                slurm_job_id=own,admission_sha256=admission_sha,prior_stage_journal_sha256=prior,
                launcher_sha256=sha(root/SELF),parallel_policy_sha256=spec['parallel_policy_sha256'],
                budget_authority_sha256=APPROVAL_SHA,historical_unjournaled_outcomes_resolved=False,**extra)
            durable_append(attempt/'execution.jsonl',row,protocol.canonical)
            durable_append(history,row,protocol.canonical)
        log('started',verification=checked);started_sha=sha(history);credit=None
        try:
            require_headroom(root,checked['cost'])
            files.durable_write(attempt/'prior.json',(json.dumps(snapshot(target,protocol))+'\n').encode())
            credit=Credit(root,f'fresh-variants-{rid}-{own}',{'admission_sha256':admission_sha,'run_id':rid})
            result=private_component(credit,history,started_sha)(root,target,rid,inference,variants,protocol,attempt/'inference')
            reconciled=reconcile(credit,target,protocol)
            files.durable_write(attempt/'credit.json',(json.dumps(dict(raw=reconciled,credit=credit.summary()))+'\n').encode())
            if not result['inference_complete']:raise RuntimeError('Variants failed after model drain')
            verify(root,rid)
            if sha(admission_path)!=admission_sha or sha(history)!=started_sha:
                raise ValueError('Admission/history changed before report')
            report=finish_report(root,target,rid,variants,protocol,analysis)
            log('complete',report_sha256=report)
        except BaseException as exc:
            if credit is not None and not credit.closed:
                if isinstance(exc,OSError):credit.journal_io_failed=True
                try:reconcile(credit,target,protocol)
                except BaseException as error:log('credit_retained',exception_type=type(error).__name__,credit=credit.summary())
            log('failed',exception_type=type(exc).__name__)
            raise
    return 0


def execute(root,rid,admission_path):
    root=root.resolve();target=replica_path(root,rid);spec=read(admission_path);own=os.environ.get('SLURM_JOB_ID')
    validate(spec,own,rid,target)
    if any(m in sys.modules for m in ('run','protocol','expanded_variants','analyze','orchestrate')):
        raise ValueError('Fresh Python process required')
    replica=verify(root,rid);sys.path.insert(0,str(target/'src'))
    import run, protocol, expanded_variants, analyze, orchestrate
    if any(m.PROJECT!=target for m in (run,expanded_variants,analyze,orchestrate)):
        raise ValueError('Mixed replica modules')
    orchestrate.verify()
    return execute_bound(root,target,rid,admission_path,spec,own,replica,run,expanded_variants,protocol,analyze)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--run-id',type=int,choices=(2,3),required=True);ap.add_argument('--admission',type=Path,required=True)
    args=ap.parse_args();raise SystemExit(execute(args.root,args.run_id,args.admission.resolve()))
