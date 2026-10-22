"""One reviewed re-admission after four pre-inference gate failures.

Keep the active variant job and all previous admissions/policies/journals.
No stage reset and no job release. New preflights required before release.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from authorized_recovery_200_reviewed import SELF, make_wrapper
from authorized_recovery_200 import APPROVAL, APPROVAL_SHA, authority, sha
from resume_schema_once import terminal_steps

ROOT=Path('/ceph/aasteine/decision-em-20261013')
FAILED={'confirmation':'366106','schema':'366107','core_run02':'366108','core_run03':'366109'}


def run(*args):return subprocess.check_output(args,text=True,env={**os.environ,'TZ':'UTC'})
def read(p):return json.loads((ROOT/p).read_text())
def save(p,obj):
    with (ROOT/p).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')


def main():
    authority(ROOT)
    oldplan=read('artifacts/operations/recovery200_held_plan_20261015.json')
    proofs={}
    for tag,job in FAILED.items():
        spec=read(oldplan['admissions'][tag])
        stage=spec['stage'];n=spec['run_id'];w=make_wrapper(ROOT,stage)
        acct=run('sacct','-n','-j',job,'--format=JobID,State,ExitCode,End','-P');terminal_steps(acct,job)
        target=ROOT if n==1 else ROOT/f'replications/run{n:02d}'
        if (target/w.ATTEMPT).exists():raise ValueError('Stage attempt exists; not an entry-gate-only failure')
        d=read(spec['diagnosis'])
        if n==1:
            inventory=w.inventory(ROOT,d)
            expected=d['files_sha256']
            for p,h in {**d['source_sha256'],**d['frozen_artifact_sha256']}.items():
                if sha(ROOT/p)!=h:raise ValueError('Prior source/journal changed')
        else:
            inventory={str(p.relative_to(target)):sha(p) for p in (target/'runs').rglob('*.jsonl')}
            expected=d['response_files_sha256']
            if sha(target/'artifacts/stage_execution.jsonl')!=d['stage_journal_sha256']:
                raise ValueError('Prior core stage journal changed')
            if json.loads((target/'artifacts/progress.json').read_text())!=d['progress']:
                raise ValueError('Prior core progress changed')
        if inventory!=expected:raise ValueError('Cache changed in pre-inference gate failure')
        log=ROOT/f'logs/{tag}-drained200-{job}.log'
        proofs[tag]={'job':job,'accounting_utc':acct,'stage_attempt_absent':True,
                     'unchanged_diagnosed_journals':len(inventory),'diagnosis_sha256':sha(ROOT/spec['diagnosis']),
                     'log_sha256':sha(log),'log_tail':log.read_text().splitlines()[-4:]}
    save('artifacts/verification/recovery200_gate_failure_review_20261015.json',
         {'utc':datetime.now(timezone.utc).isoformat(),'failures':proofs,'active_variant_job_unchanged':'366105',
          'no_inference_or_stage_journal_by_four_gate_failures':True})
    journal=ROOT/'artifacts/operations/recovery200_reviewed_submission_20261015.jsonl'
    with journal.open('x') as handle:
        def log(event,**kw):
            handle.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'event':event,**kw})+'\n');handle.flush();os.fsync(handle.fileno())
        jobs={'variants':'366105'};specs={}
        for tag in FAILED:
            spec=read(oldplan['admissions'][tag]);stage=spec['stage'];n=spec['run_id']
            path=f'configs/{tag}_budget200_reviewed_admission_20261015.json'
            script=f'artifacts/operations/{tag}_budget200_reviewed_20261015.sbatch'
            with (ROOT/script).open('x') as f:
                f.write(f'#!/bin/bash\nset -euo pipefail\ncd {ROOT}\nexec .venv/bin/python -u {SELF} --root {ROOT} --stage {stage} --run-id {n} --admission {ROOT/path}\n')
            log('submit_start',tag=tag,script=script,script_sha256=sha(ROOT/script))
            job=run('sbatch','--parsable','--hold','--partition=cpu','--cpus-per-task=2','--mem=8G',
                    '--time=2-00:00:00','--job-name=decision-em-'+tag+'-reviewed200','--chdir='+str(ROOT),
                    '--output='+str(ROOT/f'logs/{tag}-reviewed200-%j.log'),str(ROOT/script)).strip().split(';')[0]
            if not job.isdigit():raise ValueError('Uncertain submission outcome')
            log('submitted_held',tag=tag,job=job)
            jobs[tag]=job
            spec.update(slurm_job_id=job,budget_binding_sha256=sha(ROOT/SELF),
                        parallel_policy='configs/parallel_budget200_reviewed_20261015.json',
                        entry_gate_predecessor=FAILED[tag],entry_gate_review='artifacts/verification/recovery200_gate_failure_review_20261015.json',
                        entry_gate_review_sha256=sha(ROOT/'artifacts/verification/recovery200_gate_failure_review_20261015.json'))
            spec['source_sha256'][SELF]=sha(ROOT/SELF)
            specs[tag]=(path,spec)
        config=read('configs/expanded_execution.json')
        save('artifacts/operations/before_reviewed200_expanded_execution.json',config)
        config.update(e2_confirmation_job=int(jobs['confirmation']),e2_schema_job=int(jobs['schema']))
        (ROOT/'configs/expanded_execution.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
        run('scontrol','update','JobId=365300','Dependency=afterok:'+jobs['confirmation']+':'+jobs['schema'])
        run('scontrol','update','JobId=365355','Dependency=afterok:'+':'.join([jobs['confirmation'],'366105',jobs['schema'],'365300']))
        if 'Priority=0' not in run('scontrol','show','job','365300','-o').split():raise ValueError('Legacy rules hold lost')
        old=read('configs/parallel_budget200_20261015.json')
        policy={'approval_file':APPROVAL,'approval_sha256':APPROVAL_SHA,'project_cap_usd':200,
                'reviewed_project_jobs':{},'files_sha256':dict(old['files_sha256']),'legacy_rules_365300_held':True}
        policy['files_sha256'][SELF]=sha(ROOT/SELF)
        deps={}
        for line in run('squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%Z').splitlines():
            job,name,cwd=line.split('|',2)
            if cwd!=str(ROOT):continue
            if job not in set(jobs.values())|{'365300','365355'}:raise ValueError('Queue changed')
            meta=run('scontrol','show','job',job,'-o')
            deps[job]=next(s.split('=',1)[1] for s in meta.split() if s.startswith('Dependency='))
            policy['reviewed_project_jobs'][job]={'name':name,'scope':next((t for t,j in jobs.items() if j==job),'held legacy rules or waiting statistics'),
                'batch_script_sha256':hashlib.sha256(run('scontrol','write','batch_script',job,'-').encode()).hexdigest()}
        save('configs/parallel_budget200_reviewed_20261015.json',policy)
        prior_scopes=read(oldplan['admissions']['confirmation'])['reviewed_job_write_cells']
        scopes={jobs[tag]:prior_scopes[oldplan['jobs'][tag]] for tag in jobs}
        for tag,(path,spec) in specs.items():
            spec.update(parallel_policy_sha256=sha(ROOT/spec['parallel_policy']),
                        reviewed_dependencies=deps,reviewed_job_write_cells=scopes)
            save(path,spec)
        save('artifacts/operations/recovery200_reviewed_held_plan_20261015.json',
             {'jobs':jobs,'new_jobs':{t:jobs[t] for t in FAILED},'admissions':{t:p for t,(p,_) in specs.items()},
              'dependencies':deps,'policy_sha256':sha(ROOT/'configs/parallel_budget200_reviewed_20261015.json'),
              'no_old_stage_journals_reset':True})
        log('prepared_held',jobs=jobs,dependencies=deps)
        print(json.dumps({'jobs':jobs,'dependencies':deps}))


if __name__=='__main__':main()
