"""Prepare exact fixed-repeat evidence, submit one held job, then separate release."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import same_condition_launch_20261016 as launch

SELF = 'artifacts/operations/admit_same_condition_20261016.py'


def authority_budget(root):
    with launch.locks(root, exclusive=True):
        launch.queue(root)
        launch.authority(root)
        path = root/'artifacts/budget.json'
        before = launch.read(path)
        launch.require('project_cap_usd' not in before, 'Ceiling transition already attempted')
        launch.save(root/'artifacts/verification/budget250_before_20261016.json', before)
        with launch.old_budget.ledger(root) as obj:
            launch.require(obj == before, 'Ledger changed during ceiling transition')
            obj.update(project_cap_usd=250, budget_authority_sha256=launch.sha(root/launch.AUTH))
        after = launch.read(path)
        launch.require({k:v for k,v in after.items() if k not in ('project_cap_usd','budget_authority_sha256')} == before,
                       'Financial values changed during authority transition')
        launch.save(root/'artifacts/verification/budget250_transition_20261016.json',
            {'utc':launch.now(),'before_sha256':launch.sha(root/'artifacts/verification/budget250_before_20261016.json'),
             'after_sha256':launch.sha(path),'approval_sha256':launch.sha(root/launch.AUTH),
             'prior_ceiling':200,'new_ceiling':250,'spent_reserved_credits_unchanged':True})


def prepare(root):
    import core3_committed_history_20261016 as core3
    import audit_grouped_core2_terminal_20261016 as core2
    with launch.locks(root):
        launch.queue(root); budget = launch.budget_check(root, 16)
        launch.require(launch.sha(root/'artifacts/operations/same_condition_repetition_runtime_20261016.py') == launch.RUNTIME_SHA, 'Runtime changed')
        c3 = core3.verify(root, launch.ANCHOR_SHA)
        target = root/'replications/run02'
        diagnosis = launch.read(root/'artifacts/operations/grouped_core2_terminal_366696_20261016.json')
        proof = launch.read(root/'artifacts/verification/grouped_core2_terminal_366696_20261016.json')
        launch.require(launch.sha(root/'artifacts/operations/grouped_core2_terminal_366696_20261016.json') == proof['diagnosis_sha256']
            and proof['history_and_admission_verified'] and proof['credit_must_remain_held']
            and proof['unresolved_attempts'] == 1 and proof['recovery_admitted'] is False, 'Core2 terminal provenance')
        events, bound, ends, historical = core2.history(root,target)
        launch.require(events == diagnosis['stage_history'] and ends == diagnosis['all_historical_model_journals'], 'Core2 whole history drift')
        for rel, h in diagnosis['protected_sha256_or_absent'].items():
            launch.require(launch.file_state(target/rel) == h, 'Core2 old proof drift: '+rel)
        priors = {}
        for rid in (2,3):
            child = subprocess.run([sys.executable,str(root/launch.SELF),'--root',str(root),'--action','inventory-unlocked','--run-id',str(rid)],capture_output=True,text=True,check=True)
            priors[str(rid)] = json.loads(child.stdout)
            launch.require(sum(c['missing'] for c in priors[str(rid)]['cells']) == 262400, 'Unexpected fixed repetition matrix')
        jobs = list(dict.fromkeys([e['slurm_job_id'] for e in events] + list(core3.HISTORY) + ['366694','366695','367184']))
        history = launch.steps(root,jobs)
        # No new semantics from a failed core: its exact bytes are only preserved.
        protected = {}
        original_raw = {}
        sources = {launch.SELF,SELF,launch.AUTH,launch.BUDGET_POLICY,launch.BATCH,
            'configs/same_condition_repetition_costs_20261016.json',
            'artifacts/verification/same_condition_runtime_preparation_20261016.json',
            'artifacts/verification/grouped_core2_terminal_366696_20261016.json',
            'artifacts/operations/grouped_core2_terminal_366696_20261016.json',
            'artifacts/verification/pplx_repeat_terminal_367184_20261016.json'}
        sources.update(str(p.relative_to(root)) for p in (root/'artifacts/operations').glob('*.py'))
        sources.update(str(p.relative_to(root)) for p in (root/'src').glob('*.py'))
        sources.update(str(p.relative_to(root)) for p in (root/'configs').glob('*.json'))
        for rid in (2,3):
            t=root/f'replications/run{rid:02d}'; allowed=launch.own_scope(root,rid)
            for absolute,h in priors[str(rid)]['protected_sha256'].items():
                sources.add(str(Path(absolute).relative_to(root)))
            original_raw[str(rid)]={str(p.relative_to(root)):launch.sha(p) for p in (t/'runs').rglob('*.jsonl')}
            protected.update({p:h for p,h in original_raw[str(rid)].items() if p not in allowed})
            for folder in ('artifacts','configs','src'):
                for p in (t/folder).rglob('*'):
                    if p.is_file() and p.suffix != '.pyc' and p.name not in ('budget.json','parallel_writer.lock') and not p.is_symlink():
                        protected[str(p.relative_to(root))]=launch.sha(p)
        old = launch.read(root/'configs/pplx_repeat_admission_20261016.json')['protected_sha256_or_absent']
        protected.update(old)
        # PPLX protected list includes core raw/report/progress; scoped journals use append-only prior instead.
        for rel in launch.own_scope(root,2) | launch.own_scope(root,3):
            protected.pop(rel,None)
        for folder in ('output/pdf','artifacts/verification/core3_commit_anchor_367108_20261016'):
            for p in (root/folder).rglob('*'):
                if p.is_file(): protected[str(p.relative_to(root))]=launch.sha(p)
        for rel,h in historical.items(): sources.add(rel)
        costs=launch.read(root/'configs/same_condition_repetition_costs_20261016.json')
        inventory_path=root/costs['source_inventory']
        launch.require(launch.sha(inventory_path)==costs['source_inventory_sha256'], 'Historical cost provenance drift')
        sources.add(costs['source_inventory'])
        prep=dict(utc=launch.now(),source_sha256={p:launch.sha(root/p) for p in sorted(sources)},
            protected_sha256_or_absent=protected,original_raw=original_raw,priors=priors,
            history_steps=history,budget_before=budget,core3_anchor_verification=c3,
            core2_history_jobs=[e['slurm_job_id'] for e in events],
            historical_known_scenario_usd=16.253379056,future_error_costs_not_guaranteed=True,
            total_missing=524800,run1_unmodified=True,new_selection=False,calibration_calls=0,
            core2_gap_retry_allowed=False,initial_policy_bytes_available=False)
        launch.protected(root,prep); launch.foreign_credits(root,prep)
        launch.save(root/launch.PREP,prep)
        print(json.dumps({'status':'readiness_prepared_not_admitted','sources':len(sources),'protected':len(protected),'missing':524800}))


def log(root,event,**data):
    with (root/launch.JOURNAL).open('a') as f:
        f.write(json.dumps({'utc':launch.now(),'event':event,**data})+'\n');f.flush();os.fsync(f.fileno())


def full_gate(root,kind):
    p=root/f'artifacts/verification/same_condition_{kind}_20261016.json'
    subprocess.run([sys.executable,str(root/launch.SELF),'--root',str(root),'--action','preflight','--output',str(p)],check=True)
    log(root,kind+'_passed',sha256=launch.sha(p),admission_sha256=launch.sha(root/launch.ADMISSION))


def submit(root):
    with launch.locks(root):
        prep=launch.read(root/launch.PREP); launch.protected(root,prep);launch.foreign_credits(root,prep)
        jobs=launch.queue(root);launch.steps(root,prep['history_steps'],prep['history_steps']);launch.budget_check(root,16)
        launch.require(not (root/launch.JOURNAL).exists() and not (root/launch.ADMISSION).exists(), 'Already submitted')
        for rid in (2,3):
            launch.require(not (root/f'replications/run{rid:02d}'/launch.ATTEMPT).exists(),'Attempt exists')
        with (root/launch.JOURNAL).open('x') as f:f.flush();os.fsync(f.fileno())
        runtime_dir=root/launch.JOURNAL;launch.runtime.sync_dir(runtime_dir.parent)
        log(root,'held_submission_intent',batch_sha256=launch.sha(root/launch.BATCH))
        job=launch.cmd('sbatch','--hold','--parsable','--partition=cpu','--cpus-per-task=4','--mem=24G','--time=2-00:00:00',
            '--job-name='+launch.NAME,'--output='+str(root/'logs/same-condition-%j.log'),'--chdir='+str(root),str(root/launch.BATCH)).strip()
        launch.require(job.isdigit(),'Unknown submission result; do not retry')
        log(root,'held_submission_returned',job=job)
        jobs[job]={'name':launch.NAME,'batch_script_sha256':launch.sha(root/launch.BATCH),'dependency':'(null)'}
        launch.save(root/launch.POLICY,dict(jobs=jobs,scope=launch.runtime.STAGE,run_ids=[2,3],overlap_cells=0))
        spec=dict(admitted=True,scope=launch.runtime.STAGE,run_ids=[2,3],max_invocations_per_run=1,project_cap_usd=250,
            new_selection=False,calibration_calls=0,slurm_job_id=job,admission_path=launch.ADMISSION,
            readiness_sha256=launch.sha(root/launch.PREP),policy_sha256=launch.sha(root/launch.POLICY),
            approval_sha256=launch.sha(root/launch.AUTH),total_missing=524800)
        launch.save(root/launch.ADMISSION,spec)
        launch.save(root/'artifacts/operations/same_condition_job_20261016.json',dict(job=job,admission=launch.ADMISSION))
    full_gate(root,'held_gate')
    print(json.dumps({'job':job,'held':True}))


def release(root):
    events=[json.loads(x) for x in (root/launch.JOURNAL).read_text().splitlines()]
    launch.require(events[-1]['event']=='held_gate_passed' and events[-1]['admission_sha256']==launch.sha(root/launch.ADMISSION),'Held gate missing')
    full_gate(root,'prerelease_gate')
    spec=launch.read(root/launch.ADMISSION)
    log(root,'release_intent',job=spec['slurm_job_id'])
    subprocess.run(['scontrol','release',spec['slurm_job_id']],check=True)
    log(root,'release_returned',metadata=launch.cmd('scontrol','show','job',spec['slurm_job_id'],'-o'))
    print(json.dumps({'job':spec['slurm_job_id'],'released':True}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('action',choices=['budget','prepare','submit','release'])
    a=p.parse_args();{'budget':authority_budget,'prepare':prepare,'submit':submit,'release':release}[a.action](a.root.resolve())
