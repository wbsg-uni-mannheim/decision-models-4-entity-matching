"""One-shot held-job preparation after Aaron's explicit USD200 authorization.

Does not release jobs. Real read-only entry-point preflights and their review
are required separately before any scheduler release. Preserve all old files.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from authorized_recovery_200 import APPROVAL, APPROVAL_SHA, SELF, authority, make_wrapper, sha

ROOT = Path('/ceph/aasteine/decision-em-20261013')
PLAN = [
 ('variants',1,'365279','variants_terminal_diagnosis_20261015T0003Z.json','variant_remaining_cost_plan_20261015.json',None),
 ('confirmation',1,'365636','confirmation_terminal_diagnosis_20261015T0233Z.json','confirmation_remaining_cost_plan_20261015.json','confirmation_remaining_cost_verification_20261015.json'),
 ('schema',1,'365969','schema_terminal_diagnosis_20261015T0133Z.json','schema_remaining_cost_plan_20261015.json','schema_remaining_cost_verification_20261015.json'),
 ('core',2,'365893','run02_terminal_diagnosis_20261015T0103Z.json','core_run02_remaining_phase_cost_plan_20261015.json','core_phase_cost_verification_20261015.json'),
 ('core',3,'365895','run03_terminal_diagnosis_20261014T2333Z.json','core_run03_remaining_phase_cost_plan_20261015.json','core_phase_cost_verification_20261015.json')]


def run(*args):
    return subprocess.check_output(args,text=True,env={**os.environ,'TZ':'UTC'})


def read(rel):
    return json.loads((ROOT/rel).read_text())


def save(rel,obj):
    with (ROOT/rel).open('x') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def main():
    authority(ROOT)
    queue=run('squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%Z')
    project=[line.split('|')[0] for line in queue.splitlines() if str(ROOT) in line]
    if set(project)!={'365300','365355'}:
        raise ValueError('Unexpected existing project jobs; re-review required')
    legacy_meta = run('scontrol','show','job','365300','-o').split()
    if not {'Priority=0', 'JobState=PENDING', 'StartTime=Unknown'} <= set(legacy_meta):
        raise ValueError('Legacy rule job must remain held')
    for _,_,pred,*_ in PLAN:
        from resume_schema_once import terminal_steps
        terminal_steps(run('sacct','-n','-j',pred,'--format=JobID,State,ExitCode,End','-P'),pred)
    journal=ROOT/'artifacts/operations/recovery200_submission_20261015.jsonl'
    with journal.open('x') as handle:
        def log(event,**kw):
            handle.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'event':event,**kw})+'\n')
            handle.flush();os.fsync(handle.fileno())
        log('start',approval_sha256=APPROVAL_SHA,budget=read('artifacts/budget.json'))
        for rel in ('configs/expanded_execution.json','configs/parallel_execution_20261014.json'):
            dest='artifacts/operations/before_budget200_'+Path(rel).name
            save(dest,read(rel))
        jobs={};specs={};scopes={}
        for stage,n,pred,diagnosis,cost,verification in PLAN:
            tag=stage if n==1 else f'core_run{n:02d}'
            admission=f'configs/{tag}_budget200_admission_20261015.json'
            script=f'artifacts/operations/{tag}_budget200_20261015.sbatch'
            command=f'exec .venv/bin/python -u {SELF} --root {ROOT} --stage {stage} --run-id {n} --admission {ROOT/admission}'
            with (ROOT/script).open('x') as f:
                f.write(f'#!/bin/bash\nset -euo pipefail\ncd {ROOT}\n{command}\n')
            log('submit_start',tag=tag,script=script,script_sha256=sha(ROOT/script))
            job=run('sbatch','--parsable','--hold','--partition=cpu','--cpus-per-task=2','--mem=8G',
                '--time=2-00:00:00','--job-name=decision-em-'+tag+'-drained200',
                '--chdir='+str(ROOT),'--output='+str(ROOT/f'logs/{tag}-drained200-%j.log'),str(ROOT/script)).strip().split(';')[0]
            if not job.isdigit():raise ValueError('Unknown submission outcome')
            jobs[tag]=job;log('submitted_held',tag=tag,job=job,admission=admission)
            wrapper=make_wrapper(ROOT,stage)
            dp='artifacts/operations/'+diagnosis;cp='artifacts/verification/'+cost
            d=read(dp)
            spec={'admitted':True,'run_id':n,'stage':stage,'predecessor':pred,'slurm_job_id':job,
                  'budget_cap_usd':200,'budget_authority':APPROVAL,'budget_authority_sha256':APPROVAL_SHA,
                  'budget_binding_sha256':sha(ROOT/SELF),'max_invocations':1,
                  'historical_unknown_outcomes_disclosed':True,'unobserved_worker_causes_disclosed':True,
                  'unobserved_worker_causes':['kev','luna','pplx'],
                  'source_sha256':{p:sha(ROOT/p) for p in wrapper.REQUIRED},
                  'diagnosis':dp,'diagnosis_sha256':sha(ROOT/dp),'cost_plan':cp,'cost_plan_sha256':sha(ROOT/cp),
                  'parallel_policy':'configs/parallel_budget200_20261015.json',
                  'planning_caveat':'Historical scenarios are not hard bounds or a fully funded completion promise; every request shares the enforced USD200 ledger.'}
            if verification:spec['cost_source_verification']='artifacts/verification/'+verification
            if stage=='core':
                replica=read(f'replications/run{n:02d}/replica.json')
                spec.update(replica=str(ROOT/f'replications/run{n:02d}'),runner_sha256=replica['runner_sha256'])
                # Covers every prospective core namespace/condition; isolated
                # replica prefix prevents falsely merging distinct fresh runs.
                import sys
                sys.path.insert(0,str(ROOT/'src'))
                from protocol import MODELS,BENCHMARKS
                scopes[job]=[[f'replications/run{n:02d}/'+ns,m,v,b,s]
                    for ns in ('quality','timing_serial','timing_concurrent8') for m in MODELS
                    for v in ('B0','P1','P2','S1','O1') for b in BENCHMARKS for s in ('prompt','route','test')]
            else:
                scopes[job]=[[c.get('namespace','quality'),c['model'],c.get('variant','B0'),c['benchmark'],c['split']]
                             for c in d['cells'] if c['missing']>0]
                if stage in ('confirmation','schema'):
                    tests={(c['model'],c['benchmark']) for c in d['cells'] if c['split']=='test'}
                    spec['control_files_sha256']={str(p.relative_to(ROOT)):sha(p) for m,b in tests
                        for p in (ROOT/'runs/quality'/m/'B0'/b).glob('test*.jsonl')}
            specs[tag]=(admission,spec)
        for a,sa in scopes.items():
            for b,sb in scopes.items():
                if a!=b and set(map(tuple,sa))&set(map(tuple,sb)):
                    raise ValueError('Overlapping fresh writers')
        config=read('configs/expanded_execution.json')
        config.update(e6_variants_job=int(jobs['variants']),e2_confirmation_job=int(jobs['confirmation']),
                      e2_schema_job=int(jobs['schema']))
        (ROOT/'configs/expanded_execution.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
        run('scontrol','update','JobId=365300','Dependency=afterok:'+jobs['confirmation']+':'+jobs['schema'])
        run('scontrol','update','JobId=365355','Dependency=afterok:'+':'.join([jobs['confirmation'],jobs['variants'],jobs['schema'],'365300']))
        queue=run('squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%Z')
        policy={'approval_file':APPROVAL,'approval_sha256':APPROVAL_SHA,'project_cap_usd':200,
                'reviewed_project_jobs':{},'files_sha256':{},'legacy_rules_365300_held':True}
        deps={}
        for line in queue.splitlines():
            job,name,cwd=line.split('|',2)
            if cwd!=str(ROOT):continue
            if job not in set(jobs.values())|{'365300','365355'}:raise ValueError('Queue changed')
            meta=run('scontrol','show','job',job,'-o')
            deps[job]=next(s.split('=',1)[1] for s in meta.split() if s.startswith('Dependency='))
            script=run('scontrol','write','batch_script',job,'-')
            import hashlib
            policy['reviewed_project_jobs'][job]={'name':name,'batch_script_sha256':hashlib.sha256(script.encode()).hexdigest(),
                'scope': next((tag for tag,j in jobs.items() if j==job),'legacy rules held or cache-only statistics waiting')}
        for _,spec in specs.values():policy['files_sha256'].update(spec['source_sha256'])
        policy['files_sha256'].update(config['source_sha256'])
        save('configs/parallel_budget200_20261015.json',policy)
        for tag,(admission,spec) in specs.items():
            spec.update(parallel_policy_sha256=sha(ROOT/spec['parallel_policy']),reviewed_dependencies=deps,
                        reviewed_job_write_cells=scopes)
            save(admission,spec)
        save('artifacts/operations/recovery200_held_plan_20261015.json',{'jobs':jobs,'admissions':{t:p for t,(p,_) in specs.items()},
            'policy_sha256':sha(ROOT/'configs/parallel_budget200_20261015.json'),'dependencies':deps,
            'scope_intersections':0,'all_jobs_held':True,'budget_before':read('artifacts/budget.json')})
        log('prepared_held',jobs=jobs,dependencies=deps,policy_sha256=sha(ROOT/'configs/parallel_budget200_20261015.json'))
        print(json.dumps({'jobs':jobs,'dependencies':deps}))


if __name__=='__main__':main()
