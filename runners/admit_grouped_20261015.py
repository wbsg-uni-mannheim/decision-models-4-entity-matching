"""Bind the four already-held successor IDs to audited sources and joins. No release."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/ceph/aasteine/decision-em-20261013')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(*args):return subprocess.check_output(args,text=True)
def save(p,obj):
    with p.open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


def main():
    jobs=read(ROOT/'artifacts/operations/grouped_jobs_20261015.json')
    previous={'confirmation':'366671','schema':'366672','core2':'366680','core3':'366681'}
    for stage in jobs:
        diagnosis=ROOT/f'artifacts/operations/grouped_audit_{stage}_20261015.json'
        proof=read(ROOT/f'artifacts/verification/grouped_transition_{stage}_20261015.json')
        assert proof['diagnosis_sha256']==sha(diagnosis)
        assert 'Priority=0' in command('scontrol','show','job',jobs[stage],'-o')
    old=read(ROOT/'configs/parallel_fast_core_directory_20261015.json')
    for rel,h in old['files_sha256'].items():assert sha(ROOT/rel)==h,rel
    config_path=ROOT/'configs/expanded_execution.json';config=read(config_path)
    save(ROOT/'artifacts/operations/pre_grouped_expanded_execution_20261015.json',config)
    assert str(config['e2_confirmation_job'])==previous['confirmation'] and str(config['e2_schema_job'])==previous['schema']
    command('scontrol','update','JobId=365300','Dependency=afterok:'+jobs['confirmation']+':'+jobs['schema'])
    command('scontrol','update','JobId=365355','Dependency=afterok:'+jobs['confirmation']+':'+jobs['schema']+':365300')
    config.update(e2_confirmation_job=int(jobs['confirmation']),e2_schema_job=int(jobs['schema']),
        grouped_journal_execution={'jobs':jobs,'cap_usd':200,'credit_chunk_usd':8,'checkpoint_every_attempts':512,
            'frozen_experiments_unchanged':True,'original_timing_account_preserved':True,
            'user_authority':'artifacts/operations/approval_grouped_writes_aaron_20261015.json'})
    tmp=config_path.with_suffix('.fast.tmp');tmp.write_text(json.dumps(config,indent=2)+'\n');os.replace(tmp,config_path)
    policy={**old,'reviewed_project_jobs':{},'files_sha256':dict(old['files_sha256'])}
    extra=['artifacts/operations/grouped_resume_20261015.py','artifacts/operations/leased_budget.py',
        'artifacts/operations/grouped_transition.py','artifacts/verification/verify_grouped_transition_20261015.py',
        'artifacts/operations/test_fast_resume_20261015.py','artifacts/operations/test_leased_budget.py',
        'artifacts/operations/approval_grouped_writes_aaron_20261015.json','configs/expanded_execution.json',
        'artifacts/verification/io_profile_live_20261015.json','artifacts/verification/grouped_tests_remote_20261015.log']
    extra += ['artifacts/operations/grouped_inference.py','artifacts/operations/test_grouped_inference.py','artifacts/operations/test_grouped_transition.py','artifacts/operations/grouped_stop_20261015.json','artifacts/verification/grouped_stop_reconciled_20261015.json']
    extra += [f'artifacts/operations/grouped_{s}_20261015.sbatch' for s in jobs]
    for rel in extra:policy['files_sha256'][rel]=sha(ROOT/rel)
    for stage,jid in {**jobs,'rules':'365300','statistics':'365355'}.items():
        meta=command('scontrol','show','job',jid,'-o');fields=dict(x.split('=',1) for x in meta.split() if '=' in x)
        assert fields['WorkDir']==str(ROOT)
        script=command('scontrol','write','batch_script',jid,'-')
        digest=hashlib.sha256(script.encode()).hexdigest()
        if jid in ('365300','365355'):assert digest==old['reviewed_project_jobs'][jid]['batch_script_sha256']
        else:assert digest==sha(ROOT/f'artifacts/operations/grouped_{stage}_20261015.sbatch')
        policy['reviewed_project_jobs'][jid]={'name':fields['JobName'],'scope':stage,'batch_script_sha256':digest}
    policy_rel='configs/parallel_grouped_20261015.json';save(ROOT/policy_rel,policy)
    for stage,jid in jobs.items():
        dp=f'artifacts/operations/grouped_audit_{stage}_20261015.json'
        vp=f'artifacts/verification/grouped_transition_{stage}_20261015.json'
        spec={'admitted':True,'slurm_job_id':jid,'stage':stage,'predecessor':previous[stage],
            'cap_usd':200,'max_invocations':1,'new_jobs':jobs,'sources':policy['files_sha256'],
            'diagnosis':dp,'diagnosis_sha256':sha(ROOT/dp),'independent_proof':vp,'independent_proof_sha256':sha(ROOT/vp),
            'policy':policy_rel,'policy_sha256':sha(ROOT/policy_rel),
            'user_authority':'artifacts/operations/approval_grouped_writes_aaron_20261015.json',
            'remaining_cost_scenario':read(ROOT/vp)['remaining_scenario'],
            'budget_policy':'User-authorized deferred accounting with capped shared credits; full scenario is not a funding guarantee',
            'writer_scope':str(ROOT/f'replications/run0{stage[-1]}') if stage.startswith('core') else str(ROOT/'runs'/('schema' if stage=='schema' else 'quality'))}
        save(ROOT/f'configs/grouped_{stage}_admission_20261015.json',spec)
    print(json.dumps({'jobs':jobs,'policy_sha256':sha(ROOT/policy_rel),'released':False}))


if __name__=='__main__':main()
