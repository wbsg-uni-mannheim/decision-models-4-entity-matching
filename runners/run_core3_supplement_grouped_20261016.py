"""First Run3 supplement after the exact committed 367108 core.

Privately binds the finished grouped wrapper; no mutation of its source or
module globals. A separate real admission, held job and two full preflights
are required. This file never submits/releases a job or retries an attempt.
"""
import argparse
from contextlib import ExitStack
import fcntl
from pathlib import Path
import subprocess
import types

import core3_committed_history_20261016 as history
import run_fresh_supplement_grouped as original
from rules_drained import bind

SELF='artifacts/operations/run_core3_supplement_grouped_20261016.py'
ATTEMPT='artifacts/operations/core3_supplement_grouped_367108_v1'
REQUIRED=original.REQUIRED|{SELF,history.SELF,history.ANCHOR+'/manifest.json',
    history.ANCHOR+'/progress.json',history.ANCHOR+'/stage_execution.jsonl'}
ORIGINAL_SHA='55d36bca8faf8ba961f660d918781f7e4034521694592e85d70c597ae795591d'
HELPER_SHA='5db738b3f00c7bbf7085bc8d474f34f67be80d3b1d21ed6a4e4664c822bab33f'


def validate(spec,own,rid,target):
    expected=dict(admitted=True,stage='supplement',run_id=3,slurm_job_id=own,replica=str(target),
        budget_cap_usd=200,budget_authority_sha256=original.APPROVAL_SHA,max_invocations=1,
        historical_unknown_outcomes_disclosed=True,accounting_transport='bounded_process_credit',
        journal_transport='group_commit_v1',credit_chunk_usd=8,checkpoint_reservations=512,
        history_adapter=history.ADAPTER,core_job_id='367108',
        core_stage_journal_sha256=history.STAGE_SHA,progress_sha256=history.PROGRESS_SHA)
    history.require(type(rid) is int and rid==3 and type(spec.get('run_id')) is int
        and isinstance(own,str) and own.isdigit() and own not in history.HISTORY
        and all(spec.get(k)==v for k,v in expected.items()), 'Actual separately reviewed Run3 supplement admission required')
    history.require(REQUIRED <= spec.get('source_sha256',{}).keys(), 'Executable/anchor provenance incomplete')
    history.require(spec.get('core_commit_anchor_sha256') == spec['source_sha256'][history.ANCHOR+'/manifest.json'], 'Anchor pin mismatch')


def sources(root,spec):
    original.sources(root,spec)
    history.require(original.sha(root/original.SELF)==ORIGINAL_SHA
        and original.sha(root/'artifacts/operations/fresh_supplement_grouped.py')==HELPER_SHA,
        'Finished wrapper/helper changed')


def prerequisites(root,target,rid,spec,protocol,started_sha=None):
    history.require(type(rid) is int and rid==3 and target.resolve()==(root/history.TARGET).resolve(), 'Wrong replica')
    history.require(not (target/original.ATTEMPT).exists(), 'Older grouped supplement attempt exists')
    started=None
    if started_sha:
        history.require(original.sha(target/history.STAGE)==started_sha,'Current stage changed')
        attempt=target/ATTEMPT
        events=original.old.rows(attempt/'execution.jsonl')
        history.require(len(events)==1,'Only first active supplement start accepted')
        started=events[0]
        expected=dict(stage='supplement',status='started',run_id=3,slurm_job_id=spec['slurm_job_id'],
            launcher_sha256=original.sha(root/SELF),parallel_policy_sha256=spec['parallel_policy_sha256'],
            prior_cache_sha256=original.sha(attempt/'prior.json'),prior_stage_journal_sha256=history.STAGE_SHA)
        history.require(all(started.get(k)==v for k,v in expected.items()),'Wrong own started event')
        admission=history.own(root,spec['admission_path'])
        history.require(original.read(admission)==spec and started['admission_sha256']==original.sha(admission), 'Own admission provenance changed')
    return history.verify(root,spec['core_commit_anchor_sha256'],started=started)


def current_scheduler(root,target,rid,spec,own):
    result=original.common.current_scheduler(root,target,rid,spec,own)
    history.require({'365300','365355',own} <= result['dependencies'].keys(), 'Missing mandatory current queue joins')
    for job,ids in [('365300',{'366694','366695'}),('365355',{'366694','366695','365300'})]:
        raw=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        fields=dict(x.split('=',1) for x in raw.split() if '=' in x)
        dependencies=fields.get('Dependency','').split(',')
        history.require(all(v.startswith('afterok:') for v in dependencies)
            and {v.split(':')[1].split('(')[0] for v in dependencies}==ids
            and fields.get('JobState')=='PENDING' and fields.get('StartTime')=='Unknown'
            and (job!='365300' or fields.get('Priority')=='0'), 'Blocked legacy job changed')
    return result


def private_wrapper():
    """Shared originals stay untouched; each bound function sees this exact adapter."""
    common=types.SimpleNamespace(**{**vars(original.common),'current_scheduler':current_scheduler})
    updates=dict(SELF=SELF,ATTEMPT=ATTEMPT,REQUIRED=REQUIRED,validate=validate,sources=sources,
        prerequisites=prerequisites,common=common)
    updates['verify_ready']=bind(original.verify_ready,**updates)
    updates['finish_report']=bind(original.finish_report,**updates)
    updates['execute_bound']=bind(original.execute_bound,**updates)
    updates['execute']=bind(original.execute,**updates)
    return types.SimpleNamespace(**updates)


def preflight_bound(root,target,rid,admission_path,spec,own,replica,
                    inference,supplement,protocol,analysis,orchestration,*,output):
    validate(spec,own,rid,target)
    history.require(admission_path.resolve()==history.own(root,spec['admission_path']).resolve(), 'Wrong admission path')
    with ExitStack() as stack:
        for p,mode in ((root/'artifacts/operations/repetition_writer.lock',fcntl.LOCK_SH),
                       (target/'artifacts/parallel_writer.lock',fcntl.LOCK_EX)):
            f=stack.enter_context(p.open('a'));fcntl.flock(f,mode|fcntl.LOCK_NB)
        history.require(original.read(admission_path)==spec,'Admission changed')
        result=private_wrapper().verify_ready(root,target,rid,spec,own,replica,supplement,protocol)
        history.require(original.read(admission_path)==spec,'Admission changed during gate')
        value=dict(status='full_preflight_passed',admission_sha256=original.sha(admission_path),
            slurm_job_id=own,run_id=rid,verification=result,inference_started=False)
        original.files.durable_write(output,(protocol.canonical(value)+'\n').encode())
    return 0


def execute(root,admission,preflight_output=None):
    root=root.resolve();admission=admission.resolve()
    spec=original.read(admission)
    history.require(admission==history.own(root,spec.get('admission_path','')).resolve(), 'Own admission path required')
    wrapped=private_wrapper()
    if preflight_output:
        def preflight(*args):return preflight_bound(*args,output=preflight_output)
        run=bind(original.execute,SELF=SELF,validate=validate,execute_bound=preflight)
    else:run=wrapped.execute
    return run(root,3,admission)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--admission',type=Path,required=True);ap.add_argument('--preflight-output',type=Path)
    args=ap.parse_args();raise SystemExit(execute(args.root,args.admission,args.preflight_output))
