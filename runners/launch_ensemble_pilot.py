"""Scheduler admission after the existing chain; no interactive approval fabrication."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from ensemble_pilot_run import run
from selective_ensemble_costs import quote

ROOT=Path('/ceph/aasteine/decision-em-20261013')
CHAIN=['365636','365221','365268','365279','365288','365300']
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config=json.loads((ROOT/'configs/selective_ensemble_pilot_execution.json').read_text())
    for rel,expected in config['files_sha256'].items():
        if sha(ROOT/rel)!=expected:
            raise ValueError('Execution source or approval changed: '+rel)
    package=ROOT/'artifacts/strong_ensemble/pilot_v1'
    approval=json.loads((package/'approval_aaron_selective_20261014.json').read_text())
    if approval['approved_by']!='Aaron' or approval['max_cost_usd']!=10 or approval['project_cap_usd']!=100:
        raise ValueError('Wrong explicit pilot approval')
    own=os.environ['SLURM_JOB_ID']
    if config.get('parallel_policy'):
        from parallel_admission import review
        policy_path=ROOT/config['parallel_policy']
        if sha(policy_path)!=config['parallel_policy_sha256']:
            raise ValueError('Parallel policy changed')
        evidence=review(ROOT,policy_path,own)
        reviewed=evidence['reviewed_jobs']
        accounting='Independent output scopes reviewed under '+evidence['policy_sha256']
    else:
        queue=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j'],text=True)
        reviewed=[]
        for line in queue.splitlines():
            job,name=line.split('|',1)
            if job==own:
                continue
            # Full statistics is cache-only and independently documented; never an inference writer.
            if job=='365355' and name=='statistics-full':
                reviewed.append({'id':job,'role':'documented cache-only statistics'})
                continue
            meta=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
            if job in CHAIN or str(ROOT) in meta or name.startswith('decision-em'):
                raise ValueError('Existing project writer or queued inference: '+job)
            reviewed.append({'id':job,'role':'different project; scheduler metadata checked','metadata_sha256':hashlib.sha256(meta.encode()).hexdigest()})
        accounting=subprocess.check_output(['sacct','-n','-P','-j',','.join(CHAIN),'--format=JobIDRaw,State'],text=True)
        found=set()
        for line in accounting.splitlines():
            if not line.strip(): continue
            job,state=line.split('|')[:2]
            main_id=job.split('.')[0]
            if main_id not in CHAIN: continue
            found.add(main_id)
            if state.split()[0].rstrip('+') not in TERMINAL:
                raise ValueError('Prior writer/step not terminal: '+job+' '+state)
        if found!=set(CHAIN):
            raise ValueError('Incomplete accounting evidence for prior chain')
    admission={'package_sha256':approval['package_sha256'],'broad_ensemble_authorization_verified':True,
        'explicit_pilot_approval_sha256':sha(package/'approval_aaron_selective_20261014.json'),
        'todo_sha256':sha(ROOT/'docs/APPROVED_TODO_20261014.md'),
        'runner_sha256':sha(ROOT/'artifacts/operations/ensemble_pilot_run.py'),
        'all_project_inference_jobs_reviewed':True,'scheduler_job':own,'reviewed_other_jobs':reviewed,
        'prior_chain_accounting':accounting,'authorization_note':'Actual direct user message approves the described 48-request technical pilot; full strong-model evaluations prohibited.'}
    prior=package/config['prior_pilot_journal'] if config.get('prior_pilot_journal') else None
    if prior is not None:
        admission['prior_journal_sha256']=sha(prior)
    with (package/'admission.json').open('x') as f:
        json.dump(admission,f,indent=2)
    sys.path.insert(0,str(ROOT/'src'))
    import run as frozen_run
    if prior is None:
        run(ROOT,package,admission,frozen_run.key(),[])
    else:
        run(ROOT,package,admission,frozen_run.key(),[],prior_journal=prior)
    quote(ROOT/'artifacts/strong_ensemble/selective_routing_run01.json',package,
          ROOT/'artifacts/strong_ensemble/selective_cost_projection_after_pilot.json')


if __name__=='__main__': main()
