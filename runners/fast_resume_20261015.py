"""One admitted, audited resume using bounded credits. Frozen experiments intact."""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types

import authorized_recovery_200 as binding
import authorized_recovery_200_reviewed as reviewed
from drained_inference import build, durable_append
from fast_transition_audit import JOBS, locks, read, sha
from leased_budget import Credit, utc
from resume_schema_once import terminal_steps
from resume_variants_second_drained import afterok_ids

SELF='artifacts/operations/fast_resume_20261015.py'


def preflight(root, spec, own):
    stage=spec['stage']; run_id=int(stage[-1]) if stage.startswith('core') else 1
    target=root/f'replications/run{run_id:02d}' if run_id>1 else root
    if (spec.get('admitted') is not True or spec.get('slurm_job_id')!=own or not own
            or spec.get('predecessor')!=JOBS[stage] or spec.get('cap_usd')!=200
            or spec.get('max_invocations')!=1): raise ValueError('Real bounded admission required')
    binding.authority(root)
    if read(root/spec['user_authority'])['project_cap_usd']!=200: raise ValueError('Wrong accounting authority')
    for rel,h in spec['sources'].items():
        if sha(root/rel)!=h: raise ValueError('Source drift '+rel)
    diagnosis=read(root/spec['diagnosis'])
    if sha(root/spec['diagnosis'])!=spec['diagnosis_sha256'] or diagnosis['job']!=JOBS[stage] or not diagnosis['all_steps_terminal']:
        raise ValueError('Fresh terminal audit required')
    proof=read(root/spec['independent_proof'])
    if sha(root/spec['independent_proof'])!=spec['independent_proof_sha256'] or proof['diagnosis_sha256']!=spec['diagnosis_sha256'] or proof['totals']!=diagnosis['totals'] or not proof['history_and_admission_verified']:
        raise ValueError('Independent raw and history proof required')
    for rel,h in diagnosis['protected_sha256_or_absent'].items():
        # Job IDs live here; bind the new version in sources, retain audited old hash.
        if run_id==1 and rel=='configs/expanded_execution.json': continue
        path=target/rel
        if (sha(path) if path.exists() else None)!=h: raise ValueError('Audited cache/history drift '+rel)
    observed={str(p.relative_to(target)) for p in (target/'runs').rglob('*.jsonl')}
    if run_id>1 and not observed <= diagnosis['protected_sha256_or_absent'].keys(): raise ValueError('Extra replica journal')
    for job in JOBS.values():
        ac=subprocess.check_output(['sacct','-n','-j',job,'--format=JobID,State,ExitCode,End','-P'],text=True,env={**os.environ,'TZ':'UTC'})
        terminal_steps(ac,job)
    policy=binding.private_module(root,'parallel_admission',[("!= 100","!= 200",1)])
    checked=policy.review(root,root/spec['policy'],own)
    if checked['policy_sha256']!=spec['policy_sha256'] or not any(j.get('self') for j in checked['reviewed_jobs']): raise ValueError('Actual scheduler admission missing')
    config=read(root/'configs/expanded_execution.json')
    confirmation,schema=str(config['e2_confirmation_job']),str(config['e2_schema_job'])
    for job in ('365300','365355'):
        meta=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        fields=dict(x.split('=',1) for x in meta.split() if '=' in x)
        expected={confirmation,schema} | ({'365300'} if job=='365355' else set())
        if afterok_ids(fields['Dependency'])!=expected: raise ValueError('Full join mismatch')
        if job=='365300' and (fields['Priority']!='0' or fields['StartTime']!='Unknown' or fields['JobState']!='PENDING'): raise ValueError('Legacy rules must remain held')
    # Both Run1 writers are disjoint namespaces; replicas have distinct roots.
    allowed=spec['new_jobs']
    if set(allowed)!=set(JOBS) or allowed[stage]!=own: raise ValueError('Scope admission incomplete')
    if confirmation!=allowed['confirmation'] or schema!=allowed['schema']: raise ValueError('Wrong configured successor')
    if len(set(allowed.values()))!=4: raise ValueError('Overlapping job identities')
    for j in checked['reviewed_jobs']:
        if j.get('scope')!='different project' and j['id'] not in {*allowed.values(),'365300','365355'}: raise ValueError('Unexpected writer')
    variant=subprocess.check_output(['sacct','-n','-j','366563','--format=JobID,State,ExitCode,End','-P'],text=True)
    terminal_steps(variant,'366563')
    if any(x.split('|')[1:3]!=['COMPLETED','0:0'] for x in variant.splitlines() if x.strip()): raise ValueError('Variant prerequisite lost')
    budget=read(root/'artifacts/budget.json')
    if any(v['active'] for v in budget.get('maintenance_barriers',{}).values()): raise ValueError('Drain barrier active')
    if budget['spent']+budget['reserved']+.05>200: raise ValueError('No budget headroom')
    return target,diagnosis,{'policy':checked,'budget':budget,'diagnosis_sha256':spec['diagnosis_sha256']}


def credit_build(credit, root):
    def bound(inference, run_id):
        isolated=types.SimpleNamespace(**{**vars(inference),'account':credit.account})
        request,cell,provenance=build(isolated,run_id)
        # Timing experiments keep the original per-call budget path and timing definition.
        if run_id>1:
            old=binding.make_wrapper(root,'core').authorized_build(inference,run_id)[1]
            original=cell
            def dispatch(*args,**kwargs):
                return (old if kwargs.get('namespace','').startswith('timing_') else original)(*args,**kwargs)
            cell=dispatch
        return request,cell,{**provenance,'accounting_transport':'bounded_process_credit','credit':credit.identity}
    return bound


def core_component(root):
    component=binding.private_module(root,'fresh_core_drained')
    source=(root/'artifacts/operations/fresh_core_drained.py').read_text()
    old="set(status) != {'smoke', 'AP1_baseline'} or status['smoke']['status'] != 'complete'\n            or status['AP1_baseline']['status'] != 'needs_attention'"
    new="not set(status) <= {'smoke','AP1_baseline','AP2_development'} or status['smoke']['status'] != 'complete'\n            or status['AP1_baseline']['status'] not in ('needs_attention','complete')\n            or ('AP2_development' in status and status['AP2_development']['status'] != 'needs_attention')"
    if source.count(old)!=1: raise ValueError('Core phase guard drift')
    exec(compile(source.replace(old,new),'<reviewed B0 or development resume>','exec'),component.__dict__)
    return component


def new_attempt_costs(target, before, namespaces=None):
    known=[];unknown=attempts=0
    for path in (target/'runs').rglob('*.jsonl'):
        if path.name.endswith(('.operations.jsonl','.executions.jsonl')): continue
        if namespaces and path.relative_to(target/'runs').parts[0] not in namespaces: continue
        if path.relative_to(target/'runs').parts[0].startswith('timing_'): continue
        rel=str(path.relative_to(target)); prior=before.get(rel)
        offset=prior['bytes'] if prior else 0
        with path.open('rb') as f:
            prefix=f.read(offset)
            import hashlib
            if prior and hashlib.sha256(prefix).hexdigest()!=prior['sha256']: raise ValueError('Prior answers overwritten')
            for line in f:
                row=json.loads(line)
                for a in row['attempts']:
                    attempts+=1
                    if a['cost'] is None: unknown+=1
                    else: known.append(a['cost'])
    return {'attempts':attempts,'unknown':unknown,'charged':math.fsum(known)+.05*unknown}


def execute(root, admission, preflight_output=None):
    spec=read(admission);stage=spec['stage'];own=os.environ.get('SLURM_JOB_ID') if not preflight_output else spec['slurm_job_id']
    with ExitStack() as stack:
        target=locks(root,stage,stack)
        target,diagnosis,checked=preflight(root,spec,own)
        if preflight_output:
            with preflight_output.open('x') as f:json.dump(checked,f,indent=2)
            return 0
        run_id=diagnosis['run_id'];sys.path.insert(0,str(target/'src'))
        import run, protocol, orchestrate, analyze
        if run.PROJECT!=target: raise ValueError('Wrong process root')
        orchestrate.verify()
        attempt=target/f'artifacts/operations/fast_resume_{stage}_20261015'
        attempt.mkdir(exist_ok=False)
        namespaces={'schema'} if stage=='schema' else {'quality'}
        before={str(p.relative_to(target)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (target/'runs').rglob('*.jsonl') if p.relative_to(target/'runs').parts[0] in namespaces}
        (attempt/'prior_journals.json').write_text(json.dumps(before,sort_keys=True)+'\n')
        history=target/'artifacts/stage_execution.jsonl' if run_id>1 else root/('artifacts/operations/'+stage+'_recovery_'+('365636' if stage=='confirmation' else '365969')+'.jsonl')
        prior=sha(history)
        def log(status,**extra):
            row={'utc':utc(),'stage':'core' if run_id>1 else stage,'status':status,'event':'fast_resume_'+status,
                'run_id':run_id,'slurm_job_id':own,'predecessor':JOBS[stage],'recovery':'leased_budget_v1',
                'admission_sha256':sha(admission),'prior_stage_journal_sha256':prior,
                'parallel_policy_sha256':spec['policy_sha256'],**extra}
            durable_append(attempt/'execution.jsonl',row,protocol.canonical);durable_append(history,row,protocol.canonical)
        log('started',verification=checked)
        credit=Credit(root,'fast-'+stage+'-'+own,{'admission_sha256':sha(admission),'run_id':run_id})
        try:
            if stage.startswith('core'):
                from repeated_runs import verify
                verify(root,run_id)
                comp=core_component(root);comp.build=credit_build(credit,root)
                result=comp.run_phases(root,target,run_id,run,orchestrate,protocol,attempt/'phases')
                from fast_transition_audit import audit
                final=audit(root,stage,require_terminal=False)
                if final['totals']['missing']: raise ValueError('Core completion requires all scoped raw answers')
                with (attempt/'final_raw_audit.json').open('x') as f:json.dump(final,f,indent=2)
            else:
                module=__import__('expanded_'+stage)
                comp=binding.private_module(root,stage+'_drained');comp.build=credit_build(credit,root)
                args=(root,module,analyze,run,protocol,attempt/'models') if stage=='confirmation' else (root,module,analyze,run,attempt/'models')
                result=comp.run_selected_tests(*args)
                if any(m['status']!='complete' for m in result['models']): raise RuntimeError('Models drained; new diagnosis required')
                if any(o['unknown_outcome_events'] for o in result['operations']): raise ValueError('Unknown operation')
                wrapper=reviewed.make_wrapper(root,stage)
                cells,_=wrapper.selected_tests(root,module,analyze)
                verify_accounting(credit,target,before)
                credit.checkpoint(close=True)
                wrapper.finish_report(root,module,analyze,protocol,cells)
            if not credit.closed:
                verify_accounting(credit,target,before)
                credit.checkpoint(close=True)
            log('complete',component=result,credit=credit.summary())
            return 0
        except BaseException as exc:
            # Ordinary drained failure can reconcile exactly; crash/unknown outcome retains credit.
            try:
                if not credit.closed:
                    verify_accounting(credit,target,before);credit.checkpoint(close=True)
            except BaseException as reconcile:
                log('credit_retained',exception_type=type(reconcile).__name__,credit=credit.summary())
            log('failed',exception_type=type(exc).__name__,credit=credit.summary())
            raise


def verify_accounting(credit,target,before):
    observed=new_attempt_costs(target,before,{'schema'} if credit.identity.startswith('fast-schema') else {'quality'})
    if (credit.outstanding or observed['attempts']!=credit.settled or credit.settled!=credit.attempts
            or observed['unknown']!=credit.unknown or not math.isclose(observed['charged'],credit.charged,abs_tol=1e-8)):
        raise ValueError('New durable raw attempts do not reconcile with credit; keep hold')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--admission',type=Path,required=True);p.add_argument('--preflight-output',type=Path)
    a=p.parse_args();raise SystemExit(execute(a.root.resolve(),a.admission.resolve(),a.preflight_output))
