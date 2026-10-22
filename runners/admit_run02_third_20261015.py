"""Submit exactly one held Run2 successor; no release or inference here."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import resume_fresh_core_run02_third_drained as second
from resume_schema_once import terminal_steps

ROOT=Path('/ceph/aasteine/decision-em-20261013')
TARGET=ROOT/'replications/run02'
POLICY='configs/parallel_core_run02_third_20261015.json'
ADMISSION='configs/core_run02_third_admission_20261015.json'
DIAG='artifacts/operations/run02_drained_terminal_diagnosis_366545_20261015.json'
PROOF='artifacts/verification/run02_drained_terminal_verification_366545_20261015.json'
COST='artifacts/verification/core_run02_third_phase_cost_plan_20261015.json'
SCRIPT='artifacts/operations/core_run02_third_20261015.sbatch'
ACTIVE={'366115','366116','366520','366563','365300','365355'}


def run(*args):
    return subprocess.check_output(args,text=True,env={**os.environ,'TZ':'UTC'})


def read(rel):return json.loads((ROOT/rel).read_text())


def save(rel,obj):
    with (ROOT/rel).open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


def fields(job):return dict(x.split('=',1) for x in run('scontrol','show','job',job,'-o').split() if '=' in x)


def main():
    second.budget.authority(ROOT)
    w=second.make_wrapper(ROOT)
    w.verify(ROOT,2)
    for j in (second.INITIAL,'366117',second.PREDECESSOR):
        terminal_steps(run('sacct','-n','-j',j,'--format=JobID,State,ExitCode,End','-P'),j)
    d,p=read(DIAG),read(PROOF)
    assert second.budget.sha(ROOT/DIAG)=='373504b17bd63af6074f6aea06c256fe8e390dd2bae5d4668a3fc3864bb73660'
    assert second.budget.sha(ROOT/PROOF)=='cb598190c585ceb714f036f036394dd6c8d3abcf29d6ef14478567f532e87d17'
    assert p['job']==second.PREDECESSOR and p['independent_cells']==168 and p['independent_counts']==d['totals']
    assert p['diagnosis_sha256']==second.budget.sha(ROOT/DIAG)
    for rel,h in d['protected_sha256_or_absent'].items():
        assert (second.budget.sha(TARGET/rel) if (TARGET/rel).exists() else None)==h,rel
    assert not (TARGET/second.ATTEMPT).exists()
    for rel in (POLICY,ADMISSION,SCRIPT):assert not (ROOT/rel).exists()
    oldpath='configs/parallel_variants_third_20261015.json'
    old=read(oldpath);checked=w.review(ROOT,ROOT/oldpath,None)
    assert {j['id'] for j in checked['reviewed_jobs'] if j.get('scope')!='different project'}==ACTIVE
    meta={j:fields(j) for j in ACTIVE}
    assert second.afterok_ids(meta['365355']['Dependency'])=={'366115','366116','366563','365300'}
    assert second.afterok_ids(meta['365300']['Dependency'])=={'366115','366116'}
    assert meta['365300']['Priority']=='0' and meta['365300']['StartTime']=='Unknown'
    prior_stage=second.rows(ROOT/'replications/run03/artifacts/core_drained_recovery_v2/execution.jsonl')
    assert prior_stage[0]['status']=='started' and prior_stage[0]['slurm_job_id']=='366520'
    assert all(meta[j]['JobState']=='RUNNING' and meta[j]['Dependency']=='(null)' for j in ACTIVE-{'365300','365355'})
    cost=dict(diagnosis_sha256=second.budget.sha(ROOT/DIAG),source_verification_sha256=second.budget.sha(ROOT/PROOF),
              quote_or_hard_bound=False,phases=p['fresh_remaining_core_scenario']['phases'],
              caveat='Known and unknown historical scenarios separate; no quote, hard bound, representative future error forecast or full-study financing.')
    w.validate_cost(cost,second.budget.sha(ROOT/DIAG),read('artifacts/budget.json'))
    preserved={rel:second.budget.sha(ROOT/rel) for rel in [oldpath,
        'configs/parallel_budget200_reviewed_20261015.json','configs/parallel_budget200_20261015.json',
        'configs/core_run02_second_admission_20261015.json','configs/core_run02_second_reviewed_admission_20261015.json','configs/parallel_core_run02_second_reviewed_20261015.json','configs/variants_third_admission_20261015.json','configs/variants_second_admission_20261015.json','configs/core_run03_second_admission_20261015.json',
        'configs/parallel_variants_second_20261015.json',
        *[f'configs/{tag}_budget200_reviewed_admission_20261015.json' for tag in ('confirmation','schema','core_run02','core_run02')]]}
    save(COST,cost)
    with (ROOT/'artifacts/operations/core_run02_third_submission_20261015.jsonl').open('x') as journal:
        def log(event,**kw):
            journal.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),event=event,**kw))+'\n');journal.flush();os.fsync(journal.fileno())
        with (ROOT/SCRIPT).open('x') as f:
            f.write(f'#!/bin/bash\nset -euo pipefail\ncd {ROOT}\nexport PYTHONPATH=src:artifacts/operations\nexec .venv/bin/python -u {second.SELF} --root {ROOT} --admission {ROOT/ADMISSION}\n')
        log('submit_start',script=SCRIPT,sha256=second.budget.sha(ROOT/SCRIPT))
        job=run('sbatch','--parsable','--hold','--partition=cpu','--cpus-per-task=2','--mem=8G',
                '--time=2-00:00:00','--job-name=decision-em-core-run02-third200','--chdir='+str(ROOT),
                '--output='+str(ROOT/'logs/core-run02-third200-%j.log'),str(ROOT/SCRIPT)).strip().split(';')[0]
        if not job.isdigit():raise ValueError('Uncertain submission; do not repeat')
        log('submitted_held',job=job)
        policy={**old,'reviewed_project_jobs':{},'files_sha256':dict(old['files_sha256'])}
        for rel in w.REQUIRED:policy['files_sha256'][rel]=second.budget.sha(ROOT/rel)
        deps={};seen=set()
        for line in run('squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%Z').splitlines():
            jid,name,cwd=line.split('|',2)
            if cwd!=str(ROOT):continue
            assert jid in ACTIVE|{job};seen.add(jid)
            digest=hashlib.sha256(run('scontrol','write','batch_script',jid,'-').encode()).hexdigest()
            if jid in ACTIVE:
                assert name==old['reviewed_project_jobs'][jid]['name'] and digest==old['reviewed_project_jobs'][jid]['batch_script_sha256']
            else:assert digest==second.budget.sha(ROOT/SCRIPT)
            policy['reviewed_project_jobs'][jid]=dict(name=name,scope='core run02 third' if jid==job else old['reviewed_project_jobs'][jid]['scope'],batch_script_sha256=digest)
            deps[jid]=fields(jid)['Dependency']
        assert seen==ACTIVE|{job}
        save(POLICY,policy)
        spec=read(second.PRIOR_ADMISSION)
        for key in ('entry_gate_predecessor','entry_gate_review','entry_gate_review_sha256','entry_gate_proof','entry_gate_proof_sha256','unobserved_worker_causes','reviewed_job_write_cells'):
            spec.pop(key,None)
        spec.update(predecessor=second.PREDECESSOR,slurm_job_id=job,diagnosis=DIAG,diagnosis_sha256=second.budget.sha(ROOT/DIAG),
            cost_plan=COST,cost_plan_sha256=second.budget.sha(ROOT/COST),cost_source_verification=PROOF,
            parallel_policy=POLICY,parallel_policy_sha256=second.budget.sha(ROOT/POLICY),
            budget_binding_sha256=second.budget.sha(ROOT/second.SELF),reviewed_dependencies=deps,
            reviewed_job_write_roots={j:[str(ROOT/'replications/run03') if j=='366520' else str(ROOT/'runs')] for j in ACTIVE-{'365300','365355'}},
            preserved_active_admissions_and_policies_sha256=preserved)
        for rel in w.REQUIRED:spec['source_sha256'][rel]=second.budget.sha(ROOT/rel)
        save(ADMISSION,spec)
        assert all(second.budget.sha(ROOT/p)==h for p,h in preserved.items())
        save('artifacts/operations/core_run02_third_held_plan_20261015.json',dict(job=job,admission=ADMISSION,policy=POLICY,
            dependencies=deps,preserved_sha256=preserved,held_not_released=True))
        log('prepared_held',job=job,admission_sha256=second.budget.sha(ROOT/ADMISSION),dependencies=deps)
        print(json.dumps(dict(job=job,held_not_released=True,dependencies=deps)))


if __name__=='__main__':main()
