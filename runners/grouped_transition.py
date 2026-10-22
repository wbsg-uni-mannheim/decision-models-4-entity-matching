"""Controlled stop at old credit checkpoints, exact terminal reconciliation.

No signals, no job cancellation, no changes to paid totals at stop request.
Old loaded Credit rejects the next checkpoint once its status is changed. All
in-flight settlements remain local and are drained by the unchanged adapter.
"""
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import types

import fast_transition_audit as old_audit
from fast_resume_20261015 import new_attempt_costs
from leased_budget import ledger, utc
from operational_journal import inspect_operations as original_inspect
from resume_schema_once import terminal_steps

JOBS={'confirmation':'366671','schema':'366672','core2':'366680','core3':'366681'}
STOP='artifacts/operations/grouped_stop_20261015.json'
PROOF='artifacts/verification/grouped_stop_reconciled_20261015.json'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    with p.open('x') as f:
        json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


def request_stop(root, authority):
    if (root/STOP).exists():raise ValueError('Stop already requested; inspect existing receipt')
    with ledger(root) as obj:
        before={};statuses={}
        for stage,job in JOBS.items():
            ident='fast-'+stage+'-'+job;row=obj['inference_credits'][ident]
            if row['status'] not in ('active','closed'):raise ValueError('Unreviewed credit status')
            before[ident]=dict(row);statuses[stage]=row['status']
        receipt={'utc':utc(),'jobs':JOBS,'authority_sha256':sha(authority),'credits_before':before,
                 'nonfinancial_checkpoint_stop':True,'no_signals_or_cancellations':True}
        save(root/STOP,receipt)
        for stage,job in JOBS.items():
            if statuses[stage]=='active':
                obj['inference_credits']['fast-'+stage+'-'+job]['status']='drain_requested_grouped'
    return receipt


def terminal(root):
    out={}
    for stage,job in JOBS.items():
        text=subprocess.check_output(['sacct','-n','-j',job,'--format=JobID,State,ExitCode,Start,End','-P'],text=True,env={**os.environ,'TZ':'UTC'})
        terminal_steps(text,job);out[stage]=text
    return out


def target_for(root,stage):
    return root/f'replications/run0{stage[-1]}' if stage.startswith('core') else root


def reconcile(root):
    accounting=terminal(root);stop=read(root/STOP);checks={};resolved={}
    with ExitStack() as stack:
        for stage in JOBS:old_audit.locks(root,stage,stack)
        for stage,job in JOBS.items():
            target=target_for(root,stage);attempt=target/f'artifacts/operations/fast_resume_{stage}_20261015'
            events=old_audit.rows(attempt/'execution.jsonl');last=events[-1]
            if last['status'] not in ('failed','complete') or last['slurm_job_id']!=job:raise ValueError('Missing final stage summary')
            summary=last['credit'];before=read(attempt/'prior_journals.json')
            observed=new_attempt_costs(target,before,{'schema'} if stage=='schema' else {'quality'})
            if (summary['outstanding'] or summary['attempts']!=summary['settled']
                    or observed['attempts']!=summary['settled'] or observed['unknown']!=summary['unknown']
                    or not math.isclose(observed['charged'],summary['charged'],abs_tol=1e-8)):
                raise ValueError('Unreconciled raw credit; retain hold')
            # Enumerate only newly appended operational events. The original
            # inspector validates their exact requests and retry joins in audit.
            for p in (target/'runs').rglob('*.operations.jsonl'):
                if p.relative_to(target/'runs').parts[0] != ('schema' if stage=='schema' else 'quality'):continue
                rel=str(p.relative_to(target));offset=before.get(rel,{}).get('bytes',0)
                with p.open('rb') as f:f.seek(offset);lines=f.read().splitlines()
                for line in lines:
                    row=json.loads(line);s=row['operational_stop']
                    if not s['outcome_unknown']:continue
                    if (s['phase']!='reserve' or s['exception_type']!='ValueError'
                            or row['error']!='Operational reserve exception: ValueError'
                            or not stop['utc']<=row['completed_at']<=last['utc']):
                        raise ValueError('Unexplained operational outcome')
                    key=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
                    resolved[key]={'stage':stage,'path':rel,'reason':'Pinned old Credit rejects operator-marked status before reserving or HTTP'}
            checks[stage]={'summary':summary,'raw':observed,'execution_sha256':sha(attempt/'execution.jsonl')}
        # Every shared amount is reconciled exactly. No foreign hold is touched.
        with ledger(root) as obj:
            for stage,job in JOBS.items():
                ident='fast-'+stage+'-'+job;row=obj['inference_credits'][ident];s=checks[stage]['summary']
                if row['status']=='closed':
                    if not s['closed'] or row['attempts']!=s['attempts'] or row['charged']!=s['charged']:raise ValueError('Closed credit mismatch')
                    continue
                prior=stop['credits_before'][ident]
                if row!={**prior,'status':'drain_requested_grouped'}:raise ValueError('Stopped credit changed')
                if s['funded']!=row['funded'] or s['charged']<row['charged'] or s['settled']<row['settled']:raise ValueError('Invalid credit delta')
                obj['spent']+=s['charged']-row['charged']
                obj['reserved']-=row['funded']-row['charged']
                obj['attempts']+=s['attempts']-row['attempts']
                obj['unknown_cost_attempts']+=s['unknown']-row['unknown']
                row.update(status='closed',charged=s['charged'],attempts=s['attempts'],settled=s['settled'],
                    unknown=s['unknown'],outstanding_at_checkpoint=0,checkpoint_at=utc(),
                    reconciled_by=PROOF)
            if obj['reserved'] < -1e-7:raise ValueError('Negative reserve')
            budget_after={k:obj[k] for k in ('spent','reserved','attempts','unknown_cost_attempts')}
        proof={'utc':utc(),'all_steps_terminal':True,'sacct':accounting,'jobs':JOBS,
               'stop_sha256':sha(root/STOP),'credits':checks,'resolved_reserve_events':resolved,
               'budget_after':budget_after,'no_raw_records_changed':True}
        save(root/PROOF,proof)
    return proof


def inspector(root):
    proof=read(root/PROOF)
    if not proof['all_steps_terminal'] or proof['stop_sha256']!=sha(root/STOP):raise ValueError('Stop provenance changed')
    allowed=proof['resolved_reserve_events']
    def inspect(path,*args):
        result=original_inspect(path,*args)
        known=0
        for row in old_audit.rows(path):
            if row['operational_stop']['outcome_unknown']:
                key=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
                if key in allowed:
                    if not str(path).endswith('/'+allowed[key]['path']):raise ValueError('Foreign stop journal')
                    known+=1
        return {**result,'unknown_outcome_events':result['unknown_outcome_events']-known,
                'audited_controlled_reserve_stops':known,'original_unknown_outcome_events':result['unknown_outcome_events']}
    return inspect


def audit(root,stage,require_terminal=True):
    source=Path(old_audit.__file__).read_text()
    source=source.replace("attempt = (target/'artifacts/core_drained_recovery_v3') if stage.startswith('core') else root/f'artifacts/operations/{stage}_drained_recovery_v1'",
        "attempt = target/f'artifacts/operations/"+("fast" if require_terminal else "grouped")+"_resume_{stage}_20261015'")
    module=types.ModuleType('grouped_terminal_audit');module.__file__=__file__
    exec(compile(source,'<grouped terminal audit>','exec'),module.__dict__)
    module.JOBS=JOBS;module.inspect_operations=inspector(root)
    result=module.audit(root,stage,require_terminal)
    target=target_for(root,stage)
    for p in (target/f'artifacts/operations/fast_resume_{stage}_20261015').rglob('*'):
        if p.is_file():result['protected_sha256_or_absent'][str(p.relative_to(target))]=sha(p)
    result['controlled_stop_proof_sha256']=sha(root/PROOF)
    return result
